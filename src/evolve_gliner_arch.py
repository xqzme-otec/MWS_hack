"""Эволюция архитектуры GLiNER (GPU): каждый кандидат = короткое обучение по бюджету времени →
span-F1 на golden (чистый + ASR-шум 0.3) и задержка на CPU. Победитель дообучается полностью,
по нему считаются спаны для эволюции сборки (src/run_evolve_gcore.py).

Гены архитектуры: базовый энкодер (mT5-small / mDeBERTa-base), сколько слоёв энкодера оставить, LSTM над
энкодером, макс. ширина спана, dropout голов, заморозка нижних слоёв, контекст следующей реплики;
гены обучения: lr энкодера и голов, доля отрицательных реплик, число и уровень шумовых копий.
Фитнес = 0.5·(F1_чистый + F1_шум) − 0.02·max(0, задержка_с − 1)  (мягкий штраф за задержку).

  python src/evolve_gliner_arch.py --pop 6 --gens 3 --cand_minutes 9 --final_minutes 50
Продолжает с места остановки (runs/arch/history.json).
"""
import os, sys, json, random, subprocess, time, math, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--pop", type=int, default=6)
ap.add_argument("--gens", type=int, default=3)
ap.add_argument("--elite", type=int, default=2)
ap.add_argument("--cand_minutes", type=float, default=9)
ap.add_argument("--final_minutes", type=float, default=50)
ap.add_argument("--lat_pen", type=float, default=0.02)
ap.add_argument("--lat_scale", default="auto", help="перевод задержки этого CPU в эталонный: число или auto (замер x-small)")
ap.add_argument("--extra_final", default="", help="дистиллированные звонки (jsonl) — в полное обучение победителя")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dir", default="runs/arch")
ap.add_argument("--extra", default="", help="доп. аргументы обучения (отладка)")
ap.add_argument("--local_hf", default="", help="папка с локальными весами (отладка)")
a = ap.parse_args()
os.makedirs(a.dir, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
if a.lat_scale == "auto":
    cal = f"{a.dir}/calibration.json"
    if os.path.exists(cal):
        a.lat_scale = json.load(open(cal))["scale"]
    else:
        # отдельный процесс: модель калибровки не должна занимать память во время обучения кандидатов
        code = ("import sys, json; sys.path.insert(0, 'src'); from cfx.gliner_hub import calibrate; "
                f"sc, ms = calibrate(); json.dump({{'scale': sc, 'ms': ms}}, open('{cal}', 'w'))")
        subprocess.run([sys.executable, "-c", code], check=True)
        a.lat_scale = json.load(open(cal))["scale"]
    print(f"калибровка CPU: этот процессор ×{1 / a.lat_scale:.2f} от эталонного (lat_scale={a.lat_scale:.3f})", flush=True)
a.lat_scale = float(a.lat_scale)
if a.local_hf:
    BASES_LOCAL = {"xs": f"{a.local_hf}/gliner-x-small", "multi": f"{a.local_hf}/gliner_multi-v2.1"}
HIST = f"{a.dir}/history.json"
rng = random.Random(a.seed)

BASES = {"xs": ("knowledgator/gliner-x-small", 8, 8), "multi": ("urchade/gliner_multi-v2.1", 12, 8)}  # (HF, слоёв, batch)
SPEC = {
    "base": ("choice", ["xs", "multi"]),
    "keep": ("int", 3, 12),
    "rnn": ("choice", [0, 1]),
    "max_width": ("choice", [6, 8, 12]),
    "dropout": ("float", 0.1, 0.4),
    "freeze": ("int", 0, 3),
    "ctx": ("choice", [0, 10, 25]),
    "lr": ("log", 1e-5, 8e-5),
    "head_lr": ("log", 3e-5, 3e-4),
    "neg_ratio": ("float", 0.1, 0.4),
    "noise_copies": ("choice", [0, 1, 2]),
    "noise_level": ("float", 0.15, 0.45),
}
SEEDS = [  # стартовые точки: исходные архитектуры и «быстрая» гипотеза
    {"base": "xs", "keep": 8, "rnn": 1, "max_width": 12, "dropout": 0.3, "freeze": 0, "ctx": 25, "lr": 2e-5, "head_lr": 1e-4,
     "neg_ratio": 0.2, "noise_copies": 1, "noise_level": 0.3},
    {"base": "multi", "keep": 12, "rnn": 1, "max_width": 12, "dropout": 0.3, "freeze": 0, "ctx": 25, "lr": 2e-5, "head_lr": 1e-4,
     "neg_ratio": 0.2, "noise_copies": 1, "noise_level": 0.3},
    {"base": "xs", "keep": 4, "rnn": 0, "max_width": 8, "dropout": 0.2, "freeze": 0, "ctx": 0, "lr": 4e-5, "head_lr": 2e-4,
     "neg_ratio": 0.2, "noise_copies": 1, "noise_level": 0.3},
]


def sample(sp):
    if sp[0] == "choice": return rng.choice(sp[1])
    if sp[0] == "int": return rng.randint(sp[1], sp[2])
    if sp[0] == "float": return round(rng.uniform(sp[1], sp[2]), 3)
    return float(f"{math.exp(rng.uniform(math.log(sp[1]), math.log(sp[2]))):.2e}")


def fix(g):
    g = dict(g)
    g["keep"] = max(2, min(g["keep"], BASES[g["base"]][1]))
    return g


def mutate(g, rate=0.3):
    c = dict(g)
    for k, sp in SPEC.items():
        if rng.random() < rate:
            if sp[0] == "int":
                c[k] = min(sp[2], max(sp[1], c[k] + rng.choice((-2, -1, 1, 2))))
            elif sp[0] == "float":
                c[k] = round(min(sp[2], max(sp[1], c[k] + rng.gauss(0, 0.15 * (sp[2] - sp[1])))), 3)
            elif sp[0] == "log":
                c[k] = float(f"{min(sp[2], max(sp[1], c[k] * math.exp(rng.gauss(0, 0.5)))):.2e}")
            else:
                c[k] = rng.choice(sp[1])
    return fix(c)


def cross(x, y):
    return fix({k: (x if rng.random() < 0.5 else y)[k] for k in SPEC})


key = lambda g: json.dumps(fix(g), sort_keys=True)
def fit_of(f1c, f1n, lat):
    return round(0.5 * (f1c + f1n) - a.lat_pen * max(0.0, (lat or 0.0) * a.lat_scale / 1000 - 1), 4)


hist = json.load(open(HIST)) if os.path.exists(HIST) else []
for h in hist:                      # переоценка уже обученных кандидатов при новой калибровке задержки
    if h.get("fit", -1) > -1 and "f1" in h:
        h["fit"] = fit_of(h["f1"], h["f1_noise"], h["lat_ms"]); h["lat_ms_ref"] = round((h["lat_ms"] or 0) * a.lat_scale)
done = {h["key"]: h for h in hist}


def args_of(g, out, minutes, final=False):
    hf, _, bs = BASES[g["base"]]
    import torch
    if not torch.cuda.is_available():   # Mac/CPU: меньше памяти — меньше batch
        bs = max(2, bs // 2)
    hf = BASES_LOCAL[g["base"]] if a.local_hf else hf
    s = (f"python src/train_gliner_gpu.py --base {hf} --out {out} --bs {bs} --max_minutes {minutes} "
         f"--keep_layers {g['keep']} --rnn {g['rnn']} --max_width {g['max_width']} --dropout {g['dropout']} "
         f"--freeze_layers {g['freeze']} --ctx {g['ctx']} --lr {g['lr']} --head_lr {g['head_lr']} "
         f"--neg_ratio {g['neg_ratio']} --noise_copies {g['noise_copies']} --noise_level {g['noise_level']} --lat_calls 4")
    s += (" --epochs 3 --evals_per_epoch 2 --save 1" + (f" --extra {a.extra_final}" if a.extra_final else "")) if final \
        else " --epochs 1 --final_eval_only --save 0 --n_dev 80"
    return s + " " + a.extra


def run(g, tag):
    k = key(g)
    if k in done:
        return done[k]
    res_path = f"{a.dir}/{tag}.json"
    cmd = args_of(g, f"{a.dir}/tmp_model", a.cand_minutes) + f" --result {res_path}"
    print(f"\n[{time.strftime('%H:%M')}] {tag}: {json.dumps(g, ensure_ascii=False)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd + f" > {a.dir}/{tag}.log 2>&1", shell=True)
    if r.returncode != 0 or not os.path.exists(res_path):
        tail = subprocess.run(f"tail -5 {a.dir}/{tag}.log", shell=True, capture_output=True, text=True).stdout
        print("   ✗ упал:", tail.strip()[-400:], flush=True)
        rec = {"key": k, "genes": g, "tag": tag, "fit": -1.0, "error": tail[-400:]}
    else:
        R = json.load(open(res_path))
        f1c = R["best"]["dev"]["f1"] if R["best"] else 0.0
        f1n = R["best"]["dev_noise0.3"]["f1"] if R["best"] else 0.0
        lat = R["lat_ms"] or 0.0
        fit = fit_of(f1c, f1n, lat)
        rec = {"key": k, "genes": g, "tag": tag, "fit": round(fit, 4), "f1": f1c, "f1_noise": f1n, "lat_ms": lat,
               "lat_ms_ref": round(lat * a.lat_scale), "layers": R["layers"], "params_M": round(R["params"] / 1e6, 1), "minutes": round((time.time() - t0) / 60, 1)}
        print(f"   ✓ F1 {f1c:.3f} / шум {f1n:.3f}, CPU {lat:.0f} мс/звонок (на эталонном ≈{lat * a.lat_scale:.0f}) → фитнес {fit:.4f} ({rec['minutes']} мин)", flush=True)
    hist.append(rec); done[k] = rec
    json.dump(hist, open(HIST, "w"), ensure_ascii=False, indent=1)
    return rec


# ---------------- эволюция
pop = [fix(s) for s in SEEDS] + [fix({k: sample(sp) for k, sp in SPEC.items()}) for _ in range(max(0, a.pop - len(SEEDS)))]
for gen in range(a.gens):
    recs = [run(g, f"g{gen}_{i}") for i, g in enumerate(pop)]
    ranked = sorted(recs, key=lambda r: -r["fit"])
    print(f"\n=== поколение {gen}: лучший фитнес {ranked[0]['fit']} — {json.dumps(ranked[0]['genes'], ensure_ascii=False)}", flush=True)
    if gen == a.gens - 1:
        break
    elite = [r["genes"] for r in ranked[: a.elite]]
    parents = [r["genes"] for r in ranked[: max(3, a.pop // 2)] if r["fit"] > -1]
    children = []
    while len(children) < a.pop - a.elite:
        c = mutate(cross(rng.choice(parents), rng.choice(parents)))
        if key(c) not in done and key(c) not in {key(x) for x in children}:
            children.append(c)
    pop = elite + children

best = max(hist, key=lambda r: r["fit"])
json.dump(best, open(f"{a.dir}/best.json", "w"), ensure_ascii=False, indent=1)
print(f"\n### ПОБЕДИТЕЛЬ: {json.dumps(best['genes'], ensure_ascii=False)}  фитнес {best['fit']}", flush=True)

# ---------------- полное обучение победителя + спаны
if not os.path.exists("out/arch_best/train_log.json"):
    print(f"[{time.strftime('%H:%M')}] полное обучение победителя ({a.final_minutes} мин)…", flush=True)
    subprocess.run(args_of(best["genes"], "out/arch_best", a.final_minutes, final=True) +
                   f" --result {a.dir}/final.json > {a.dir}/final.log 2>&1", shell=True)
subprocess.run(f"python src/gliner_infer.py --model out/arch_best --tag arch > {a.dir}/infer.log 2>&1", shell=True)
print(f"[{time.strftime('%H:%M')}] готово", flush=True)
