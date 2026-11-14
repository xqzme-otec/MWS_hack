"""HTTP-сервис CompactFact v2: POST /extract — транскрипт с метаданными → факты в схеме кейса."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cfx.objects import refresh
from cfx.schema import validate_call

MODEL = os.environ.get("CFX_MODEL", "models/compactfact_v5.pkl")
GENES = os.environ.get("CFX_GENES", "models/genes_best.json")
GLINER = os.environ.get("CFX_GLINER", "")          # путь к дообученному GLiNER (опционально)
ex = pickle.load(open(MODEL, "rb")); refresh(ex.objects)
genes = json.load(open(GENES))
DENOISE = os.environ.get("CFX_DENOISE", "0") == "1"          # шумоподавитель на входе (по умолчанию выкл.: на dev не улучшил T и S)
den = pickle.load(open(os.environ.get("CFX_DENOISER", "models/denoiser.pkl"), "rb")) if DENOISE else None
gl = None
if GLINER:
    import torch
    torch.set_num_threads(int(os.environ.get("CFX_THREADS", "1")))
    from gliner import GLiNER
    from cfx.gliner_fusion import GlinerSpans
    gl = GlinerSpans(GLiNER.from_pretrained(GLINER))
app = FastAPI(title="CompactFact v2", version="2.0")


class Turn(BaseModel):
    idx: int
    speaker: str
    text: str


class Call(BaseModel):
    call_id: str
    started_at: str
    duration_sec: Optional[int] = None
    series: Optional[dict] = None
    turns: List[Turn]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "gliner": bool(gl), "denoise": DENOISE, "genes": GENES}


@app.post("/extract")
def extract(call: Call):
    c = call.model_dump()
    t = time.perf_counter()
    if den is not None:   # исправляем текст только для модели; индексы реплик сохраняются
        c = {**c, "turns": [{**tt, "text": den(tt["text"])} for tt in c["turns"]]}
    s = ex.score(c)
    if gl is not None:
        s["gl_spans"] = gl.spans(s["view"])
    out = ex.assemble(s, genes)
    errors = validate_call(out)
    if errors:
        raise HTTPException(500, {"schema_errors": errors[:5]})
    if den is not None:   # в ответе — исходные тексты реплик
        out["turns"] = call.model_dump()["turns"]
    out["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out
