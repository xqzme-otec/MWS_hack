"""HTTP-сервис CompactFact: API кейса + веб-интерфейс.

  GET  /                  веб-интерфейс
  POST /extract           звонок (call_id, started_at, turns) → факты в схеме кейса (+ meta: задержка, движок)
                          ?mode=hybrid|gliner|boosting  ?debug=1 (сырые спаны GLiNER)
  POST /api/score         звонок с эталонными facts → факты модели + метрики кейса на этом звонке
  GET  /api/examples      список звонков для демо (data/val.jsonl, если смонтирован; иначе examples/)
  GET  /api/examples/{i}  звонок целиком (с эталоном, если есть)
  GET  /api/report        сводка метрик и ресурсов (reports/summary.json)
  GET  /health
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(ROOT, "data"))
from typing import List, Optional  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from cfx.pipeline import Pipeline  # noqa: E402

MODE = os.environ.get("CFX_MODE", "hybrid")
P = Pipeline(os.environ.get("CFX_GLINER_DIR", "models/g13_ru"),
             os.environ.get("CFX_BOOST", "models/compactfact_v5.pkl"),
             os.environ.get("CFX_GENES", "models/genes_gliner_hybrid.json"), mode=MODE)
app = FastAPI(title="CompactFact · MTS MWS", version="3.0",
              description="Извлечение компактных фактов из транскриптов звонков: GLiNER + бустинг, CPU")
WEB = os.path.join(ROOT, "web")
app.mount("/static", StaticFiles(directory=WEB), name="static")


def _examples():
    paths = [os.environ["CFX_EXAMPLES"]] if os.environ.get("CFX_EXAMPLES") else []
    for path in paths + [os.path.join(ROOT, "data", "val.jsonl"), os.path.join(ROOT, "examples", "calls.jsonl")]:
        path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if os.path.exists(path):
            return [json.loads(l) for l in open(path, encoding="utf-8")], os.path.relpath(path, ROOT)
    return [], None


EXAMPLES, EX_SRC = _examples()
try:
    import case_metrics as cm
    cm.load_catalog(os.path.join(ROOT, "data", "val.jsonl"))   # каталог ищется рядом: data/catalog.json
except Exception:   # оценщик не обязателен для /extract
    cm = None


class Turn(BaseModel):
    idx: Optional[int] = None
    speaker: str
    text: str


class Call(BaseModel):
    call_id: str = "demo"
    started_at: str
    duration_sec: Optional[int] = None
    series: Optional[dict] = None
    turns: List[Turn]
    facts: Optional[list] = None


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "engine": MODE, "gliner": os.path.relpath(P.gliner_dir, ROOT) if P.glc else None,
            "examples": EX_SRC, "n_examples": len(EXAMPLES)}


def _run(call, mode, debug):
    c = call.model_dump()
    for i, t in enumerate(c["turns"]):
        t["idx"] = i if t.get("idx") is None else t["idx"]
    if not c["turns"]:
        raise HTTPException(422, "пустой звонок")
    out = P.extract(c, with_debug=debug, mode=mode)
    if out["meta"]["schema_errors"]:
        raise HTTPException(500, {"schema_errors": out["meta"]["schema_errors"][:5]})
    return c, out


@app.post("/extract")
def extract(call: Call, mode: Optional[str] = Query(None, pattern="^(hybrid|gliner|boosting)$"), debug: int = 0):
    _c, out = _run(call, mode, bool(debug))
    return out


@app.post("/api/score")
def score(call: Call, mode: Optional[str] = Query(None, pattern="^(hybrid|gliner|boosting)$")):
    c, out = _run(call, mode, True)
    res = {"prediction": out, "metrics": None, "matches": []}
    if cm is not None and c.get("facts"):
        gold = {c["call_id"]: c}
        res["metrics"] = cm.evaluate(gold, {c["call_id"]: out})
        for g, p, s in cm.match_facts(c["facts"], out["facts"]):
            res["matches"].append({"gold": c["facts"].index(g), "pred": out["facts"].index(p), "sim": round(s, 3)})
    return res


@app.get("/api/examples")
def examples():
    return {"source": EX_SRC, "items": [
        {"i": i, "call_id": c["call_id"], "started_at": c["started_at"][:10], "n_turns": len(c["turns"]),
         "n_facts": len(c.get("facts") or [])} for i, c in enumerate(EXAMPLES)]}


@app.get("/api/examples/{i}")
def example(i: int):
    if not 0 <= i < len(EXAMPLES):
        raise HTTPException(404, "нет такого звонка")
    return EXAMPLES[i]


@app.get("/api/report")
def report():
    p = os.path.join(ROOT, "reports", "summary.json")
    return JSONResponse(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {})
