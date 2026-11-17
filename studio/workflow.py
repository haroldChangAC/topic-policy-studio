"""Join every source row to its result and persist auditable exports."""
import importlib.metadata
import io
import json
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from studio.modeling import fit_topics, model_config
from studio.runtime import ROOT
from studio.segmentation import plan_segments
from studio.corpus import missing_text_formulae
from studio.visualization import default_visualization
from studio.similarity import default_similarity


def analyze(prepared, encoder, device="cpu", progress=lambda message: None, split_mode="stop", keyword_settings=None, clustering_settings=None):
    f = prepared.fields
    missing = missing_text_formulae(prepared.corpus, prepared.text_column)
    if missing:
        raise ValueError(f"正文列有 {len(missing)} 个公式没有可用缓存结果；请在 Excel 中计算并保存或替换为文本。")
    plan = plan_segments(prepared, encoder, split_mode)
    if plan.blocked:
        raise ValueError(f"{plan.counts['long_documents']} 篇有效文档超过模型 {plan.settings['effective_max_tokens']} token 上限；请选择自动切分，或先处理原文。")
    rows = plan.documents.copy(deep=True)
    segments = plan.segments.copy(deep=True)
    fit = fit_topics(segments[f["segment_text"]].tolist(), encoder, progress, keyword_settings=keyword_settings,
                     clustering_settings=clustering_settings)
    segments[f["topic_id"]] = pd.array(fit["topics"], dtype="Int64")
    segments[f["topic_keywords"]] = [fit["keywords"][topic] for topic in fit["topics"]]
    segments[f["membership_strength"]] = pd.array(
        fit["strengths"] if fit["strengths"] is not None else [pd.NA] * len(segments), dtype="Float64")
    segments[f["status"]] = "modeled"
    rows.loc[prepared.selected, f["status"]] = "modeled"
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    run_id = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    config = model_config()
    for key, value in {"run_id": run_id, "model_id": config["model_id"], "model_revision": config["revision"],
                       "duplicate_policy": prepared.duplicate_policy, "source_sha256": prepared.corpus.sha256,
                       "text_column": prepared.text_column}.items():
        rows[f[key]] = value
        segments[f[key]] = value
    metadata = {"run_id": run_id, "completed_at": now.isoformat(), "source_sha256": prepared.corpus.sha256,
                "encoding": prepared.corpus.encoding, "text_column": prepared.text_column,
                "source_format": prepared.corpus.source_format, "source_details": prepared.corpus.source_details,
                "original_columns": prepared.corpus.frame.columns.tolist(), "generated_fields": f,
                "duplicate_policy": prepared.duplicate_policy, "counts": {**prepared.counts, **plan.counts},
                "schema_version": 2, "analysis_unit": "text_segment", "segmentation": plan.settings,
                "document_topic_aggregation": "none",
                "duplicate_rule": "exact original nonempty text; extra copies after the first",
                "empty_rule": "text.strip() is empty; original text remains unmodified",
                "record_id_rule": ("SHA256 of source bytes + sheet-name hash + Excel row number" if prepared.corpus.source_format == 'xlsx'
                                   else "SHA256 of source bytes + TXT marker + record ordinal" if prepared.corpus.source_format == 'txt'
                                   else "SHA256 of source bytes + 1-based record ordinal"),
                "model_id": config["model_id"], "model_revision": config["revision"],
                "offline": True, "embedding_device": device, "clustering_device": "cpu",
                "parameters": fit["parameters"], "clustering_settings_requested": fit['clustering_settings_requested'], "keyword_processing": fit['keyword_settings'],
                "empty_keyword_topics": fit['empty_keyword_topics'], "max_input_tokens": fit["max_input_tokens"],
                "model_token_limit": fit["model_token_limit"], "timings_seconds": fit["timings_seconds"],
                "topic_count_excluding_outliers": len(set(fit["topics"]) - {-1}),
                "outlier_count": fit["topics"].count(-1),
                "outlier_share": fit["topics"].count(-1) / len(fit["topics"]),
                "membership_strength_note": "HDBSCAN membership strength, not calibrated probability or topic intensity",
                "versions": {name: importlib.metadata.version(name) for name in (
                    "bertopic", "torch", "sentence-transformers", "umap-learn", "hdbscan",
                    "scikit-learn", "pandas", "numpy", "jieba", "openpyxl", "transformers", "tokenizers")}}
    visualization = default_visualization(fit['keyword_weights'])
    visualization['topic_similarity'] = default_similarity(fit['topic_vectors'], fit['topic_vector_dimension'])
    metadata['keyword_weight_source'] = visualization['weight_source']
    result = {"rows": rows, "segments": segments, "topics": fit["topic_info"], "metadata": metadata, "visualization": visualization}
    from studio.hierarchy import ensure_hierarchy
    ensure_hierarchy(result)
    return result


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def excel_bytes(result):
    """Explicit text cells preserve leading zeros, formulas-as-text and long IDs."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    metadata = result["metadata"]
    meta = pd.DataFrame([(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
                        columns=["setting", "value_json"])
    for name, frame in (("records", result["rows"]), ("segments", result.get("segments", pd.DataFrame())),
                        ("topics", result["topics"]), ("run_metadata", meta)):
        if len(frame) > 1048575 or len(frame.columns) > 16384:
            raise ValueError("数据超出 Excel 行列容量，请缩小输入文件。")
        sheet = workbook.create_sheet(name)
        for row in [list(frame.columns), *frame.itertuples(index=False, name=None)]:
            values = []
            for value in row:
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                elif pd.isna(value):
                    value = None
                if isinstance(value, str) and (len(value) > 32767 or ILLEGAL_CHARACTERS_RE.search(value)):
                    raise ValueError("存在 Excel 无法无损保存的超长字段或控制字符，请先处理输入。")
                values.append(value)
            sheet.append(values)
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    if result.get('topic_edits', {}).get('history'):
        sheet = workbook.create_sheet('topic_name_history')
        history = result['topic_edits']['history']
        headers = list(history[0])
        sheet.append(headers)
        for item in history:
            sheet.append([item.get(key) for key in headers])
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
        sheet.freeze_panes = "A2"
    buffer = io.BytesIO()
    workbook.save(buffer)
    # The installed XML writer emits literal CR, which XML readers normalize to
    # LF. Numeric entities preserve the original code point and hence offsets.
    preserved = io.BytesIO()
    with zipfile.ZipFile(buffer) as source, zipfile.ZipFile(preserved, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.startswith("xl/worksheets/") and info.filename.endswith(".xml"):
                data = data.replace(b"\r", b"&#13;")
            target.writestr(info, data)
    return preserved.getvalue()


def persist(result, corpus, output_root: Path | None = None, *, include_exports: bool = True):
    root = output_root if output_root is not None else ROOT / "outputs/runs"
    root.mkdir(parents=True, exist_ok=True)
    run_id = result["metadata"]["run_id"]
    temporary = root / ("." + run_id)
    destination = root / run_id
    temporary.mkdir()
    try:
        (temporary / f"source.{corpus.source_format}").write_bytes(corpus.raw)
        if include_exports:
            (temporary / "records.csv").write_bytes(csv_bytes(result["rows"]))
            (temporary / "topics.csv").write_bytes(csv_bytes(result["topics"]))
            if result.get('visualization', {}).get('keyword_weights') is not None:
                weights = pd.DataFrame(result['visualization']['keyword_weights']).assign(
                    weight_source=result['visualization']['weight_source'],
                    bertopic_version=result['metadata']['versions']['bertopic'])
                (temporary / 'keyword_weights.csv').write_bytes(csv_bytes(weights))
            if "segments" in result:
                (temporary / "segments.csv").write_bytes(csv_bytes(result["segments"]))
            try:
                (temporary / "results.xlsx").write_bytes(excel_bytes(result))
            except ValueError as exc:
                # Retain full CSV and project data when Excel cannot represent a cell.
                (temporary / "excel_error.txt").write_text(str(exc), encoding="utf-8")
        (temporary / "run.json").write_text(json.dumps(result["metadata"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return destination
