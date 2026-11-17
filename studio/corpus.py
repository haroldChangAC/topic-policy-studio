"""Lossless CSV fields, deterministic row IDs and explicit selection policy."""
import csv
import hashlib
import io
from dataclasses import dataclass, field
import zipfile
from datetime import date, datetime, time
from openpyxl import load_workbook

import pandas as pd

FIELDS = ("record_id", "source_record", "original_text", "status", "duplicate_of",
          "topic_id", "topic_keywords", "membership_strength", "run_id", "model_id",
          "model_revision", "duplicate_policy", "source_sha256", "text_column",
          "document_id", "segment_id", "segment_order", "start_char", "end_char", "segment_text",
          "token_count", "segment_count", "is_long")


@dataclass
class Corpus:
    raw: bytes
    frame: pd.DataFrame
    sha256: str
    encoding: str
    source_format: str = 'csv'
    source_details: dict = field(default_factory=dict)
    source_rows: list = field(default_factory=list)


@dataclass
class Prepared:
    corpus: Corpus
    rows: pd.DataFrame
    fields: dict
    text_column: str
    duplicate_policy: str
    counts: dict

    @property
    def selected(self):
        return self.rows[self.fields["status"]].eq("pending")


def read_csv(raw: bytes, encoding: str = "utf-8-sig") -> Corpus:
    if not raw or len(raw) > 10 * 1024**2:
        raise ValueError("请选择非空且不超过 10 MiB 的 CSV 文件。")
    try:
        text = raw.decode(encoding, errors="strict")
    except (UnicodeError, LookupError) as exc:
        raise ValueError("无法按所选编码读取文件，请更换编码后重试。") from exc
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = next(reader)
        if not headers or any(not h.strip() for h in headers):
            raise ValueError("CSV 必须包含非空列名。")
        if len(headers) != len(set(headers)):
            raise ValueError("CSV 有重名列，请先改成唯一列名，避免字段丢失。")
        rows = []
        for index, row in enumerate(reader, start=1):
            # A physical blank record is retained as an all-empty record.
            if not row:
                row = [""] * len(headers)
            if len(row) != len(headers):
                raise ValueError(f"第 {index} 条数据记录有 {len(row)} 列，表头有 {len(headers)} 列。")
            rows.append(row)
    except (csv.Error, StopIteration) as exc:
        raise ValueError("CSV 格式不完整或引号不匹配。") from exc
    return Corpus(raw, pd.DataFrame(rows, columns=headers, dtype=str),
                  hashlib.sha256(raw).hexdigest(), encoding)



def _check_size(raw):
    if not raw or len(raw) > 10 * 1024**2:
        raise ValueError('文件为空或超过 10 MiB。')


def read_txt(raw: bytes, encoding: str = 'utf-8-sig') -> Corpus:
    _check_size(raw)
    try:
        text = raw.decode(encoding, errors='strict')
    except (UnicodeError, LookupError) as exc:
        raise ValueError('TXT 无法按所选编码解码，请检查编码。') from exc
    if not text.strip():
        raise ValueError('TXT 文件没有有效文本。')
    return Corpus(raw, pd.DataFrame([{'text': text}], dtype=str), hashlib.sha256(raw).hexdigest(),
                  encoding, 'txt', {'encoding': encoding, 'unit': '整个文件作为一个原始文档'}, [1])


def _open_xlsx(raw, data_only=False):
    _check_size(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if any(x.file_size > 100*1024**2 for x in infos) or sum(x.file_size for x in infos) > 200*1024**2:
                raise ValueError('XLSX 解压后过大，无法安全读取。')
        return load_workbook(io.BytesIO(raw), read_only=True, data_only=data_only, keep_links=False)
    except ValueError as exc:
        if str(exc) == 'XLSX 解压后过大，无法安全读取。':
            raise
        raise ValueError('XLSX 文件损坏或无法读取，请检查文件格式。') from exc
    except Exception as exc:
        raise ValueError('XLSX 文件损坏或无法读取，请检查文件格式。') from exc


def xlsx_sheets(raw):
    workbook = _open_xlsx(raw)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


def _cell_text(value):
    if value is None:
        return ''
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    return str(value)


def read_xlsx(raw: bytes, sheet: str) -> Corpus:
    formulas = _open_xlsx(raw, False)
    cached = None
    try:
        if sheet not in formulas.sheetnames:
            raise ValueError('所选工作表不存在。')
        cached = _open_xlsx(raw, True)
        ws, values = formulas[sheet], cached[sheet]
        if ws.max_row is None or ws.max_column is None or ws.max_row * ws.max_column > 2_000_000:
            raise ValueError('工作表范围过大或无法确定，请缩小后重试。')
        source = ws.iter_rows(); calculated = values.iter_rows()
        headers = next(source, None); next(calculated, None)
        if not headers or all(c.value is None for c in headers):
            raise ValueError('所选工作表为空或缺少表头。')
        if any(c.data_type == 'f' for c in headers):
            raise ValueError('表头不能使用公式，请改为固定列名。')
        names = [_cell_text(c.value) for c in headers]
        if any(not n.strip() for n in names) or len(set(names)) != len(names):
            raise ValueError('工作表必须有非空且不重复的表头。')
        rows, row_numbers, formula_cells, text_columns = [], [], [], set()
        for number, (left, right) in enumerate(zip(source, calculated), start=2):
            row = []
            for col, (cell, cached_cell) in enumerate(zip(left, right)):
                if cell.data_type == 'e':
                    raise ValueError(f'单元格 {cell.coordinate} 为 Excel 错误值，请先修复。')
                if cell.data_type == 'f':
                    if not isinstance(cell.value, str):
                        raise ValueError(f'单元格 {cell.coordinate} 使用尚不支持的数组公式，请转换为固定值。')
                    available = cached_cell.value is not None and cached_cell.data_type != 'e'
                    formula_cells.append({'coordinate':cell.coordinate,'row':number,'column':names[col],
                                          'formula':str(cell.value),'cached_available':available})
                    if available and isinstance(cached_cell.value, str) and cached_cell.value.strip():
                        text_columns.add(names[col])
                    row.append(_cell_text(cached_cell.value) if available else str(cell.value))
                else:
                    if isinstance(cell.value, str) and cell.value.strip():
                        text_columns.add(names[col])
                    row.append(_cell_text(cell.value))
            rows.append(row); row_numbers.append(number)
        if not rows:
            raise ValueError('工作表只有表头，没有数据行。')
        details={'sheet':sheet,'header_row':1,'formula_policy':'使用文件中缓存结果；无缓存保留公式文本，正文列含无缓存公式时停止建模；不会计算公式',
                 'formula_cells':formula_cells,'detected_text_columns':[n for n in names if n in text_columns],
                 'cell_conversion':'字符串原样；数字和布尔转文本；日期转 ISO；不模拟 Excel 显示格式'}
        return Corpus(raw, pd.DataFrame(rows, columns=names, dtype=str),hashlib.sha256(raw).hexdigest(),
                      'xlsx', 'xlsx', details, row_numbers)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('XLSX 工作表内容无法读取。') from exc
    finally:
        formulas.close()
        if cached:cached.close()


def missing_text_formulae(corpus, text_column):
    return [item for item in corpus.source_details.get('formula_cells', [])
            if item['column'] == text_column and not item['cached_available']]

def prepare(corpus: Corpus, text_column: str, duplicate_policy: str = "keep") -> Prepared:
    if text_column not in corpus.frame.columns:
        raise ValueError("请选择文件中存在的文本列。")
    if duplicate_policy not in ("keep", "deduplicate"):
        raise ValueError("未知的重复文本处理方式。")
    prefix = "studio_"
    while any(prefix + field in corpus.frame.columns for field in FIELDS):
        prefix = "_" + prefix
    fields = {field: prefix + field for field in FIELDS}
    rows = corpus.frame.copy(deep=True)
    texts = rows[text_column]
    empty = texts.str.strip().eq("")
    blocked_rows = {item['row'] for item in missing_text_formulae(corpus, text_column)}
    source_row_values = corpus.source_rows or list(range(1, len(rows) + 1))
    blocked = pd.Series([number in blocked_rows for number in source_row_values], index=rows.index)
    comparable = ~empty & ~blocked
    normalized = texts.where(comparable)
    duplicate = comparable & normalized.duplicated(keep="first")
    involved = comparable & normalized.duplicated(keep=False)
    source_rows = corpus.source_rows or list(range(1, len(rows) + 1))
    namespace = (':sheet-' + hashlib.sha256(corpus.source_details['sheet'].encode('utf-8')).hexdigest()[:16]
                 if corpus.source_format == 'xlsx' else (':txt' if corpus.source_format == 'txt' else ''))
    ids = [f"{corpus.sha256}{namespace}:{index:08d}" for index in source_rows]
    rows[fields["record_id"]] = ids
    rows[fields["source_record"]] = source_rows
    rows[fields["original_text"]] = texts.copy()
    rows[fields["status"]] = "pending"
    rows.loc[empty, fields["status"]] = "excluded_empty"
    rows.loc[blocked, fields["status"]] = "blocked_formula_unavailable"
    first = {}
    duplicate_of = []
    for record_id, text, usable in zip(ids, texts, comparable):
        duplicate_of.append(first.get(text, "") if usable else "")
        if usable:
            first.setdefault(text, record_id)
    rows[fields["duplicate_of"]] = duplicate_of
    if duplicate_policy == "deduplicate":
        rows.loc[duplicate, fields["status"]] = "excluded_duplicate"
    counts = {"input": len(rows), "empty": int(empty.sum()), "duplicate_extra": int(duplicate.sum()),
              "duplicate_involved": int(involved.sum()),
              "selected": int(rows[fields["status"]].eq("pending").sum())}
    if corpus.source_format == 'xlsx':
        counts['formula_uncached'] = int(blocked.sum())
    return Prepared(corpus, rows, fields, text_column, duplicate_policy, counts)
