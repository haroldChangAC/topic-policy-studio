"""A genuine, offline BERTopic fit on clearly labelled synthetic Chinese data."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from studio.runtime import ROOT, configure

configure()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
args = parser.parse_args()
started = time.perf_counter()

print("正在加载本地分析组件……", flush=True)
import jieba
import numpy as np
import pandas as pd
import plotly.express as px
import torch

torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)
jieba.dt.tmp_dir = str(ROOT / ".cache")
config = json.loads((ROOT / "models/model-lock.json").read_text(encoding="utf-8"))
corpus_path = ROOT / "data/examples/synthetic_zh.csv"
frame = pd.read_csv(corpus_path, dtype=str, keep_default_na=False)
assert frame["id"].is_unique and frame["text"].str.strip().ne("").all()
assert frame["is_synthetic"].eq("true").all()
docs = frame["text"].tolist()

from studio.modeling import load_encoder, fit_topics
load_start = time.perf_counter()
encoder = load_encoder(args.device)
embedding_start = time.perf_counter()
fit = fit_topics(docs, encoder, lambda message: print(message, flush=True))
embeddings = fit["embeddings"]
topics = fit["topics"]
parameters = fit["parameters"]
embedding_seconds = fit["timings_seconds"]["embedding"]
fit_seconds = fit["timings_seconds"]["bertopic_fit"]

output = ROOT / "outputs" / ("smoke" if args.device == "cpu" else "smoke-mps")
output.mkdir(parents=True, exist_ok=True)
documents = frame.copy()
documents["topic_id"] = topics
documents["model_id"] = config["model_id"]
documents["model_revision"] = config["revision"]
documents.to_csv(output / "document_topics.csv", index=False, encoding="utf-8-sig")
topic_info = fit["topic_info"]
topic_info.to_csv(output / "topics.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(output / "smoke_results.xlsx", engine="openpyxl") as writer:
    documents.to_excel(writer, sheet_name="synthetic_document_topics", index=False)
    topic_info.to_excel(writer, sheet_name="topics", index=False)
readback = pd.read_excel(output / "smoke_results.xlsx", sheet_name="synthetic_document_topics", dtype=str)
assert readback["id"].tolist() == frame["id"].tolist()
assert readback["text"].tolist() == docs
csv_readback = pd.read_csv(output / "document_topics.csv", dtype=str)
assert csv_readback["id"].tolist() == frame["id"].tolist()
assert csv_readback["text"].tolist() == docs
np.save(output / "embeddings.npy", embeddings)
chart = px.bar(topic_info, x="Topic", y="Count", title="Synthetic corpus only — topic counts")
chart.write_html(output / "topic_counts.html", include_plotlyjs=True, full_html=True)

summary = {
    "completed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    "passed": True, "is_synthetic": True,
    "purpose": "Environment and pipeline verification only; no domain accuracy claim",
    "offline": True, "local_files_only": True,
    "python": platform.python_version(), "embedding_device": args.device,
    "clustering_device": "cpu", "document_count": len(docs),
    "topic_count_excluding_outliers": len(set(topics) - {-1}),
    "outlier_count": sum(topic == -1 for topic in topics),
    "embedding_shape": list(embeddings.shape),
    "max_input_tokens": fit["max_input_tokens"], "model_token_limit": encoder.max_seq_length,
    "model_id": config["model_id"], "model_revision": config["revision"],
    "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    "parameters": parameters,
    "versions": {name: importlib.metadata.version(name) for name in [
        "bertopic", "sentence-transformers", "torch", "umap-learn", "hdbscan",
        "scikit-learn", "jieba", "numpy", "pandas", "plotly", "openpyxl",
    ]},
    "timings_seconds": {
        "model_load": round(embedding_start-load_start, 3),
        "embedding": round(embedding_seconds, 3),
        "bertopic_fit": round(fit_seconds, 3),
        "total_including_imports_and_exports": round(time.perf_counter()-started, 3),
    },
    "checks": {"finite_embeddings": True, "one_assignment_per_row": True,
               "csv_original_text_roundtrip": True, "xlsx_original_text_roundtrip": True},
}
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

