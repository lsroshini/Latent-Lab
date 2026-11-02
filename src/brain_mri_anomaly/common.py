import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = ROOT / "data/processed/prepared_v3"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cohort_fingerprint(records):
    payload=[{"dataset":r["dataset"],"scan_id":r["scan_id"],"split":r["split"],
              "version":r.get("version"),"prepared_sha256":sha256(r["prepared_path"])}
             for r in sorted(records,key=lambda r:(r["dataset"],r["scan_id"]))]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
