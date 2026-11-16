#!/usr/bin/env bash
# Полное воспроизведение обучения CompactFact с нуля (данные кейса → веса → метрики на val).
#
#   bash scripts/reproduce.sh            # всё: бустинг → GLiNER → урезание словаря → оценка
#   SMOKE=1 bash scripts/reproduce.sh    # смоук-тест конвейера за ~10 мин на CPU (крошечные объёмы)
#   EVOLVE=1 bash scripts/reproduce.sh   # + заново подобрать гены сборки эволюцией (≈1 ч CPU)
#
# Гиперпараметры — configs/*.json (найдены эволюцией, см. docs/EXPERIMENTS.md). Seed = 42.
# Время: бустинг ≈ 10 мин CPU; GLiNER ≈ 30 мин GPU T4 / Apple M-серии, ≈ 4–6 ч на 8 ядрах CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/repro}
mkdir -p "$OUT" reports
for f in data/train.jsonl data/val.jsonl; do
  [ -f "$f" ] || { echo "нет $f — скопируйте train.jsonl и val.jsonl из пакета кейса в data/"; exit 1; }
done
log() { echo -e "\n\033[1;31m▶ $*\033[0m"; }

if [ "${SMOKE:-0}" = "1" ]; then
  GL_EXTRA="--n_calls 40 --n_dev 20 --total_steps 30 --evals_per_epoch 1"; FIT_EXTRA="--smoke"; EVAL_LIMIT="--limit 15"
else
  GL_EXTRA=""; FIT_EXTRA=""; EVAL_LIMIT=""
fi

log "1/5 Бустинг-ранжировщик кандидатов (cross-fitting на train)"
python src/fit_v4.py --out "$OUT/v4.pkl" $FIT_EXTRA
BASE_GENES=configs/genes_evo2.json python src/run_current.py "$OUT/v4.pkl" "$OUT/compactfact_v5.pkl"

log "2/5 GLiNER g13: gliner-x-small → 3 слоя энкодера + BiLSTM, max_width 6 (архитектура из эволюции)"
# словарь урезается ДО обучения: строки удалённых токенов не получают градиентов (их нет в корпусе),
# Adam поэлементный → обучение оставшихся весов идентично, а памяти нужно в ~3 раза меньше (≈ 4 ГБ RAM)
BASE_DIR=$(python -c "import sys; sys.path.insert(0, 'src'); from cfx.gliner_hub import resolve_base; print(resolve_base('knowledgator/gliner-x-small'))" | tail -1)
python src/prune_vocab.py --src "$BASE_DIR" --dst "$OUT/base_ru" --corpus data/train.jsonl data/val.jsonl
ARGS=$(python - <<'EOF'
import json
d = json.load(open("configs/train_g13.json"))
flags = []
for k, v in d.items():
    if isinstance(v, bool):
        flags += [f"--{k}"] if v else []
    else:
        flags += [f"--{k}", str(v)]
print(" ".join(flags))
EOF
)
python src/train_gliner_gpu.py $ARGS --base "$OUT/base_ru" --out "$OUT/g13" $GL_EXTRA

log "3/5 Финальная упаковка весов (fp16 при хранении; словарь уже русский: 250k → ~37k токенов, 160M → ~51M параметров)"
python src/prune_vocab.py --src "$OUT/g13" --dst "$OUT/g13_ru" --corpus data/train.jsonl data/val.jsonl --fp16

if [ "${EVOLVE:-0}" = "1" ]; then
  log "3b. Эволюция генов сборки гибрида (GLiNER на всех репликах)"
  python src/gliner_infer.py --model "$OUT/g13_ru" --tag g13r --out_dir "$OUT/spans" --sets golden,dev_B,val
  python src/run_evolve_gcore.py --spans "$OUT/spans" --tag repro_hyb --model "$OUT/compactfact_v5.pkl" \
      --base configs/genes_gliner_core_seed.json --lat configs/bench_gliner_cpu.json --hybrid --no_prefilter --no_stop
  GENES=runs/repro_hyb_best.json
else
  GENES=models/genes_gliner_hybrid.json
fi

log "4/5 Оценка на val и adversarial официальным case_metrics.py"
python src/evaluate.py --sets val,val_adv --gliner "$OUT/g13_ru" --boost "$OUT/compactfact_v5.pkl" --genes "$GENES" \
    --out_dir "$OUT/eval" $EVAL_LIMIT

log "5/5 Готово. Метрики: $OUT/eval/*_metrics.json ; веса: $OUT/g13_ru, $OUT/compactfact_v5.pkl"
echo "Чтобы запустить сервис на своих весах:  CFX_GLINER_DIR=$OUT/g13_ru CFX_BOOST=$OUT/compactfact_v5.pkl docker compose up"
