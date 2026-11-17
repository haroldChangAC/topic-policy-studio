"""Check installed dependencies and hardware without downloading models."""

import importlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from studio.runtime import ROOT, configure

configure()

PACKAGES = {
    "bertopic": "bertopic", "streamlit": "streamlit", "pandas": "pandas",
    "openpyxl": "openpyxl", "sentence-transformers": "sentence_transformers",
    "torch": "torch", "umap-learn": "umap", "hdbscan": "hdbscan",
    "scikit-learn": "sklearn", "jieba": "jieba", "plotly": "plotly",
    "matplotlib": "matplotlib", "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors", "psutil": "psutil", "setuptools": None,
}

report = {
    "checked_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    "python": platform.python_version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "packages": {}, "errors": [],
}
for distribution, module in PACKAGES.items():
    start = time.perf_counter()
    try:
        print(f"Checking {distribution}…", flush=True)
        if module:
            importlib.import_module(module)
        version = importlib.metadata.version(distribution)
        report["packages"][distribution] = version
        print(f"OK {distribution} {version} ({time.perf_counter()-start:.1f}s)", flush=True)
    except Exception as exc:
        report["errors"].append(f"{distribution}: {type(exc).__name__}: {exc}")
        print(f"FAIL {report['errors'][-1]}", flush=True)

try:
    import psutil
    import torch
    report["memory_gib"] = round(psutil.virtual_memory().total / 1024**3, 2)
    report["logical_cpus"] = psutil.cpu_count()
    report["mps_built"] = torch.backends.mps.is_built()
    report["mps_available"] = torch.backends.mps.is_available()
    report["cpu_tensor_check"] = (torch.ones(2, 2) @ torch.ones(2, 2)).tolist()
    if report["mps_available"]:
        tensor = torch.ones(2, 2, device="mps")
        report["mps_tensor_check"] = (tensor @ tensor).cpu().tolist()
except Exception as exc:
    report["errors"].append(f"hardware: {type(exc).__name__}: {exc}")

report["passed"] = not report["errors"]
destination = ROOT / "outputs" / "environment_check.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "packages"}, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["passed"] else 1)

