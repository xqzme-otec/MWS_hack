"""JSONL IO and account-disjoint, deterministic train splits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def write_jsonl(path, calls):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for call in calls:
            stream.write(json.dumps(call, ensure_ascii=False) + "\n")


def account(call):
    return (call.get("series") or {}).get("account_id") or call["call_id"].split(":")[0]


def partition(call, seed=42):
    """70% representation fit, 15% ranker fit, 15% threshold development."""
    bucket = int(hashlib.sha256(f"{seed}:{account(call)}".encode()).hexdigest()[:8], 16) % 100
    return "fit" if bucket < 70 else "rank" if bucket < 85 else "dev"


def inference_call(call):
    """Deliberate allowlist: annotations can never enter feature extraction."""
    return {k: call[k] for k in ("call_id", "started_at", "turns", "series", "duration_sec", "noise_level") if k in call}