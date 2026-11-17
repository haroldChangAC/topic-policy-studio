"""Download a pinned public model; no research text is sent."""

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from studio.runtime import ROOT, configure

configure(offline=False)

from huggingface_hub import snapshot_download

config = json.loads((ROOT / "models/model-lock.json").read_text(encoding="utf-8"))
snapshot = snapshot_download(
    repo_id=config["model_id"],
    revision=config["revision"],
    allow_patterns=config["files"],
    token=False,
    max_workers=3,
)

from pathlib import Path

snapshot = Path(snapshot)
files = []
for name in config["files"]:
    path = snapshot / name
    if not path.is_file():
        raise FileNotFoundError(f"模型缺少必要文件：{name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    files.append({"name": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})

manifest = {
    "downloaded_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    "model_id": config["model_id"], "revision": config["revision"], "files": files,
}
output = ROOT / "outputs/model_download.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"模型下载完成：{config['model_id']} @ {config['revision']}")
print(f"共 {sum(item['bytes'] for item in files)/1024**2:.1f} MiB；文件校验值已记录。")

