"""Эволюция архитектуры GLiNER2.5 (metrika-labs/ru-pii-ner-gliner2.5) для фактов звонков.

Каждый кандидат = короткое обучение (--cand_steps шагов) → span-F1 на golden (чистый + ASR-шум 0.3) и задержка на CPU.
Гены: слоёв энкодера mDeBERTa (2…12), LoRA или полное дообучение и ранг LoRA, lr энкодера и голов,
пустых меток на пример (важно для счётчика GLiNER2), доля отрицательных реплик, шумовые копии и их уровень, контекст.
Словарь всегда урезается до русских токенов (−190 млн параметров «других языков»).
Фитнес = 0.5·(F1_чистый + F1_шум) − 0.02·max(0, задержка_эталон_с − 1). Победитель обучается долго, считаются спаны.

  python src/g2/evolve_g2.py --pop 6 --gens 3 --cand_steps 1500 --final_steps 8000
Продолжает с места остановки (runs/g2arch/history.json).
"""
import os, sys, json, random, subprocess, time, math, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--pop", type=int, default=6); ap.add_argument("--gens", type=int, default=3); ap.add_argument("--elite", type=int, default=2)
ap.add_argument("--cand_steps", type=int, default=1500); ap.add_argument("--final_steps", type=int, default=8000)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lat_pen", type=float, default=0.02)
ap.add_argument("--lat_scale", default="auto", help="перевод задержки этого CPU в эталонный: число или auto")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--dir", default="runs/g2arch")
ap.add_argument("--extra", default="", help="доп. аргументы обучения (отладка)")
ap.add_argument("--infer_limit", type=int, default=0, help="отладка: звонков на набор при расчёте спанов")
a = ap.parse_args()
os.makedirs(a.dir, exist_ok=True)
HIST = f"{a.dir}/history.json"
rng = random.Random(a.seed)

if a.lat_scale == "auto":
    cal = f"{a.dir}/calibration.json"
    if not os.path.exists(cal):
        code = ("import sys, json; sys.path.insert(0, 'src'); from cfx.gliner_hub import calibrate; "
                f"sc, ms = calibrate(); json.dump({{'scale': sc, 'ms': ms}}, open('{cal}', 'w'))")
        r = subprocess.run([sys.executable, "-c", code])
        if r.returncode != 0:
            json.dump({"scale": 1.0, "ms": None, "note": "калибровка не удалась"}, open(cal, "w"))
    a.lat_scale = json.load(open(cal))["scale"]
    print(f"калибровка CPU: этот процессор ×{1 / a.lat_scale:.2f} от эталонного", flush=True)
a.lat_scale = float(a.lat_scale)

SPEC = {
    "keep": ("int", 2, 12), "lora": ("choice", [0, 1]), "lora_r": ("choice", [8, 16, 32]),
    "lr": ("log", 1e-5, 1e-4), "task_lr": ("log", 1e-4, 1e-3), "neg_labels": ("choice", [1, 2, 3]),
    "neg_ratio": ("float", 0.1, 0.4), "noise_copies": ("choice", [0, 1, 2]), "noise_level": ("float", 0.15, 0.45),
    "ctx": ("choice", [0, 10]),
}
SEEDS = [
    {"keep": 12, "lora": 0, "lora_r": 16, "lr": 3e-5, "task_lr": 5e-4, "neg_labels": 2, "neg_ratio": 0.2, "noise_copies": 1, "noise_level": 0.3, "ctx": 0},
    {"keep": 6, "lora": 0, "lora_r": 16, "lr": 5e-5, "task_lr": 5e-4, "neg_labels": 2, "neg_ratio": 0.2, "noise_copies": 1, "noise_level": 0.3, "ctx": 0},
    {"keep": 3, "lora": 0, "lora_r": 16, "lr": 5e-5, "task_lr": 5e-4, "neg_labels": 2, "neg_ratio": 0.28, "noise_copies": 0, "noise_level": 0.16, "ctx": 0},
    {"keep": 6, "lora": 1, "lora_r": 16, "lr": 5e-5, "task_lr": 5e-4, "neg_labels": 2, "neg_ratio": 0.2, "noise_copies": 1, "noise_level": 0.3, "ctx": 0},
]


def sample(sp):
    if sp[0] == "choice": return rng.choice(sp[1])
    if sp[0] == "int": return rng.randint(sp[1], sp[2])
    if sp[0] == "float": return round(rng.uniform(sp[1], sp[2]), 3)
    return float(f"{math.exp(rng.uniform(math.log(sp[1]), math.log(sp[2]))):.2e}")


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
    return c


cross = lambda x, y: {k: (x if rng.random() < 0.5 else y)[k] for k in SPEC}
key = lambda g: json.dumps(g, sort_keys=True)
fit_of = lambda f1c, f1n, lat: round(0.5 * (f1c + f1n) - a.lat_pen * max(0.0, (lat or 0.0) * a.lat_scale / 1000 - 1), 4)
hist = json.load(open(HIST)) if os.path.exists(HIST) else []
done = {h["key"]: h for h in hist}


def args_of(g, out, steps, final=False):
    s = (f"{sys.executable} -u src/g2/train_g2.py --out {out} --keep_layers {g['keep']} --lora {g['lora']} --lora_r {g['lora_r']} "
         f"--lr {g['lr']} --task_lr {g['task_lr']} --neg_labels {g['neg_labels']} --neg_ratio {g['neg_ratio']} "
         f"--noise_copies {g['noise_copies']} --noise_level {g['noise_level']} --ctx {g['ctx']} --steps {steps} --bs {a.bs} --lat_calls 4")
    s += " --save 1" if final else " --save 0 --n_dev 80"
    return s + " " + a.extra


def run(g, tag):
    k = key(g)
    if k in done:
        return done[k]
    res = f"{a.dir}/{tag}.json"
    print(f"\n[{time.strftime('%H:%M')}] {tag}: {json.dumps(g, ensure_ascii=False)}", flush=True)
    t0 = time.time()
    r = subprocess.run(args_of(g, f"{a.dir}/tmp_model", a.cand_steps) + f" --result {res} > {a.dir}/{tag}.log 2>&1", shell=True)
    if r.returncode != 0 or not os.path.exists(res):
        tail = subprocess.run(f"tail -5 {a.dir}/{tag}.log", shell=True, capture_output=True, text=True).stdout
        print("   ✗ упал:", tail.strip()[-400:], flush=True)
        rec = {"key": k, "genes": g, "tag": tag, "fit": -1.0, "error": tail[-400:]}
    else:
        R = json.load(open(res))
        f1c, f1n, lat = R["dev"]["f1"], R["dev_noise0.3"]["f1"], R["lat_ms"] or 0.0
        fit = fit_of(f1c, f1n, lat)
        rec = {"key": k, "genes": g, "tag": tag, "fit": fit, "f1": f1c, "f1_noise": f1n, "lat_ms": lat,
               "lat_ms_ref": round(lat * a.lat_scale), "layers": R["layers"], "params_M": round(R["params"] / 1e6, 1),
               "minutes": round((time.time() - t0) / 60, 1)}
        print(f"   ✓ F1 {f1c:.3f} / шум {f1n:.3f}, CPU {lat:.0f} мс/звонок (эталон ≈{lat * a.lat_scale:.0f}), "
              f"{rec['params_M']}M параметров → фитнес {fit:.4f} ({rec['minutes']} мин)", flush=True)
    hist.append(rec); done[k] = rec
    json.dump(hist, open(HIST, "w"), ensure_ascii=False, indent=1)
    return rec


pop = SEEDS + [{k: sample(sp) for k, sp in SPEC.items()} for _ in range(max(0, a.pop - len(SEEDS)))]
for gen in range(a.gens):
    recs = [run(g, f"g{gen}_{i}") for i, g in enumerate(pop)]
    ranked = sorted(recs, key=lambda r: -r["fit"])
    print(f"\n=== поколение {gen}: лучший фитнес {ranked[0]['fit']} — {json.dumps(ranked[0]['genes'], ensure_ascii=False)}", flush=True)
    if gen == a.gens - 1:
        break
    elite = [r["genes"] for r in ranked[: a.elite]]
    parents = [r["genes"] for r in ranked[: max(3, a.pop // 2)] if r["fit"] > -1]
    children = []
    while len(children) < a.pop - a.elite and parents:
        c = mutate(cross(rng.choice(parents), rng.choice(parents)))
        if key(c) not in done and key(c) not in {key(x) for x in children}:
            children.append(c)
    pop = elite + children

best = max(hist, key=lambda r: r["fit"])
json.dump(best, open(f"{a.dir}/best.json", "w"), ensure_ascii=False, indent=1)
print(f"\n### ПОБЕДИТЕЛЬ: {json.dumps(best['genes'], ensure_ascii=False)}  фитнес {best['fit']}", flush=True)
if not os.path.exists("out/g2_best/weights.pt"):
    print(f"[{time.strftime('%H:%M')}] полное обучение победителя ({a.final_steps} шагов)…", flush=True)
    subprocess.run(args_of(best["genes"], "out/g2_best", a.final_steps, final=True) + f" --result {a.dir}/final.json > {a.dir}/final.log 2>&1", shell=True)
print(f"[{time.strftime('%H:%M')}] спаны…", flush=True)
subprocess.run(f"{sys.executable} -u src/g2/infer_g2.py --model out/g2_best --tag g2 --limit {a.infer_limit} > {a.dir}/infer.log 2>&1", shell=True)
print(f"[{time.strftime('%H:%M')}] готово", flush=True)
