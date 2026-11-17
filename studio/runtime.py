"""Project-local caches and opt-in model download settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(*, offline: bool = True) -> None:
    for key, path in {
        "HF_HOME": ROOT / "models" / "cache",
        "MPLCONFIGDIR": ROOT / ".cache" / "matplotlib",
        "NUMBA_CACHE_DIR": ROOT / ".cache" / "numba",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"

