"""Задержка GLiNER на CPU: архитектура × контекст × квантизация int8 × потоки (мс на реплику и на звонок)."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="xs=../hf/gliner-x-small,multi=../hf/gliner_multi-v2.1,xb=../hf/gliner-x-base")
ap.add_argument("--ctx", default="0,25"); ap.add_argument("--threads", default="1,2"); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--out", default="runs/bench_gliner_cpu.json")
a = ap.parse_args()
from gliner import GLiNER
from gliner.data_processing.tokenizer import WordsSplitter
from cfx.data import splits
from cfx.gliner_fusion import GlinerSpans
calls = splits()["golden"][: a.n]
n_turns = sum(len(c["turns"]) for c in calls)
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for spec in a.models.split(","):
    tag, path = spec.split("=")
    m = GLiNER.from_pretrained(path).float().eval()
    m.data_processor.words_splitter = WordsSplitter("whitespace"); m.config.words_splitter_type = "whitespace"
    params = sum(p.numel() for p in m.parameters())
    for quant in ("fp32", "int8"):
        mm = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8) if quant == "int8" else m
        for th in [int(x) for x in a.threads.split(",")]:
            torch.set_num_threads(th)
            for ctx in [int(x) for x in a.ctx.split(",")]:
                gl = GlinerSpans(mm, context_words=ctx, raw_threshold=0.1, batch=16)
                gl.spans(calls[0])  # прогрев
                t = time.time()
                with torch.no_grad():
                    for c in calls: gl.spans(c)
                el = time.time() - t
                k = f"{tag}|{quant}|t{th}|c{ctx}"
                res[k] = {"ms_per_turn": round(1000 * el / n_turns, 1), "ms_per_call": round(1000 * el / len(calls), 0), "params_M": round(params / 1e6, 1)}
                print(k, res[k], flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
    del m
