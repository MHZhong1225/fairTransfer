"""Paths and durable evidence files for the BAGEL detail-supervision study."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = Path('/home/ubuntu/anaconda3/envs/ucp/bin/python')
PRIOR = ROOT / 'experiments/paper_sft_replication_v2'
BAGEL_REPO = Path('/home/ubuntu/zmh/personalization/third_party/Bagel')
BAGEL_WEIGHTS = Path('/home/ubuntu/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
