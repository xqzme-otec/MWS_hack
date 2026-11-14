# HANDOFF: дистилляция 10 000 диалогов через локальную LLM

Тебе нужно поднять локально большую модель (> 17B) с OpenAI-совместимым API и прогнать через неё
10 000 синтетических звонков. Модель-учитель переписывает черновики в живую речь, код проверяет каждый
ответ. Результат — файл `runs/distilled_10k.jsonl`, на нём дообучаем экстрактор.

Всё, что ниже, запускается из папки `CompactFact_v2/` (ветка `compactfact-v2`).

---

## 1. Что уже сделано (коротко)

**Задача кейса.** Из шумных транскриптов звонков в техподдержку извлекать факты по схеме `data/schema.md`:
состояние услуги, тариф, статус заявки, действие клиента, договорённость. Для каждого факта нужно указать
объект, говорящего, даты `valid_from`/`valid_to` и реплику-источник. Критерий жюри на val (оценщик
`data/case_metrics.py` v2.1): `temporal_recall ≥ 0.4506` и `state_accuracy ≥ 0.5145`.

**Наш результат (val, официальный оценщик):**

| метрика | база glm-5.3 | наша модель | порог |
|---|---|---|---|
| temporal_recall | 0.4706 | **0.5344** | 0.4506 |
| state_accuracy | 0.5345 | **0.5746** | 0.5145 |

Критерий пройден. Модель работает на CPU: 157 мс на звонок, 344 МБ памяти, GPU не нужен.

**Как устроено** (`src/cfx/`):
1. Кандидаты значений: char-TF-IDF-прототипы × реплики.
2. Ранжировщик HistGBM на каждый атрибут.
3. Фильтр ловушек из `schema.md` §4: отказ, гипотеза, «у соседа», «оператор зачитал».
4. Выбор объекта.
5. Даты: банк из 22 типов относительных выражений плюс обучаемый выбор кандидата.
6. Селектор текущего значения.

Пороги и режимы («гены», `models/genes_best.json`) подобраны эволюцией на golden-наборе (150 звонков из
train) с перепроверкой на отложенном dev_B. val в обучении и подборе не участвовал.

**Что пробовали и отбросили:**
- GLiNER-x-small + LoRA: +0.004 к T при задержке ×20, эволюция его выключила.
- Qwen3-0.6B как экстрактор: 112 с на звонок и невалидный JSON.
- Шумоподавитель ASR: на dev не помог (по умолчанию выключен).

**Баги оценщика v2.1** (учитываем, но не эксплуатируем):
- падает с `TypeError` на «позапрошл…» и «через две недели» — исправленная копия в `src/scorer_fixed.py`;
- в state_accuracy при равных датах подтекает эталон — фитнес эволюции считается по `src/scorer_strict.py`.

**Чего не хватает — твоя часть.** Синтетика из генератора (`src/synth/generator.py`) правильная по фактам и
датам, но звучит шаблонно. LLM-учитель должен переписать 10 000 черновиков в естественную речь, сохранив
факты. Потом дообучим модель на этих данных и посмотрим, растут ли метрики.

---

## 2. Где что лежит

```
CompactFact_v2/
├── HANDOFF.md                 ← этот файл
├── README.md                  ← описание решения, быстрый старт, API
├── data/                      ← данные кейса v2.1: train/val/baseline_pred, catalog.json, schema.md, case_metrics.py
├── models/
│   ├── compactfact_v5.pkl     ← обученная модель (текущая лучшая)
│   ├── genes_best.json        ← лучшие гены (эволюция 4)
│   ├── genes_gliner_on.json   ← гены с включённым GLiNER (не нужен)
│   └── denoiser.pkl           ← шумоподавитель (не нужен)
├── src/
│   ├── cfx/                   ← модель (extractor, dates, objects, current, evolve, bench…)
│   ├── synth/
│   │   ├── generator.py       ← генератор черновиков по плану фактов (детерминированный, seed 42)
│   │   ├── markers.py         ← частоты маркеров речи (по open-stt-calls и корпусу)
│   │   ├── teacher.py         ← LangChain: промпты учителя, разбор JSON, проверка ответа
│   │   └── run_distill.py     ← ★ массовый прогон 10 000 запросов (параллельно, с продолжением)
│   ├── fit_v4.py              ← обучение экстрактора (--extra = дистиллированные данные)
│   ├── run_current.py         ← дообучение селектора текущего значения
│   ├── run_evolve.py          ← эволюция генов
│   ├── eval_val.py            ← ★ метрики golden / dev_B / val одной командой
│   └── api.py                 ← FastAPI-сервис (/extract)
├── runs/                      ← логи эволюций, метрики, предсказания, markers.json
└── deploy/                    ← requirements.txt, Dockerfile, compose.yaml
```

Адаптер GLiNER (105 МБ) в git не положен: он больше лимита GitHub и для твоей задачи не нужен.

---

## 3. Шаг за шагом

### 3.0. Окружение (Python 3.11+)

```bash
git clone https://github.com/xqzme-otec/MWS_hack.git && cd MWS_hack
git switch compactfact-v2 && cd CompactFact_v2
python -m venv .venv && source .venv/bin/activate
pip install -r deploy/requirements.txt
```

Проверка, что модель живая (≈ 35 с):

```bash
python src/eval_val.py --model models/compactfact_v5.pkl --genes models/genes_best.json --tag check --sets golden
# ожидаемо: golden temporal_recall≈0.519 state_accuracy≈0.553
```

### 3.1. Поднять учителя — модель > 17B с OpenAI-совместимым API

Подходит любой сервер, который отдаёт `/v1/chat/completions`. Нужна модель, хорошо пишущая по-русски.

**Вариант А — NVIDIA GPU, vLLM (самый быстрый, рекомендую):**

```bash
pip install vllm
# 48–80 ГБ (A6000 / A100 / H100): MoE 30B всего / 3B активных — очень быстрая, без «размышлений»
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 8192 --gpu-memory-utilization 0.92 --port 8000
# 24 ГБ (RTX 3090/4090): официальная 4-битная плотная 32B
vllm serve Qwen/Qwen3-32B-AWQ --max-model-len 6144 --gpu-memory-utilization 0.92 --port 8000
#   (или 4-битную AWQ-сборку Qwen3-30B-A3B-Instruct-2507 с HF — она в несколько раз быстрее 32B)
```

```bash
export TEACHER_BASE_URL=http://localhost:8000/v1
export TEACHER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   # ровно то имя, что в vllm serve
export TEACHER_API_KEY=local
WORKERS=32
```

**Вариант Б — Mac (Apple Silicon, ≥ 32 ГБ) или любой ПК, Ollama:**

```bash
ollama pull qwen3:30b-a3b          # или qwen3:32b, gemma3:27b
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

```bash
export TEACHER_BASE_URL=http://localhost:11434/v1
export TEACHER_MODEL=qwen3:30b-a3b
export TEACHER_API_KEY=ollama
WORKERS=4
```

LM Studio тоже подойдёт (`http://localhost:1234/v1`, имя модели — как в LM Studio).

**Сколько это займёт.** Один запрос — около 900 входных и 400–800 выходных токенов. 10 000 запросов —
примерно 5–8 млн выходных токенов.

| сервер | примерное время |
|---|---|
| vLLM, 30B-A3B, одна карта, 32 потока | 1–2 ч |
| vLLM, плотная 32B AWQ на 24 ГБ, 16 потоков | 3–6 ч |
| Ollama на Mac, 30B-A3B | 10–20 ч (можно в несколько заходов, см. ниже) |

**Режим рассуждения.** Для Qwen3 скрипт сам дописывает в запрос `/no_think` — рассуждение отключается,
так в 5–10 раз быстрее. Для других моделей это безвредно; отключить можно через `TEACHER_NO_THINK=0`.

### 3.2. Сгенерировать черновики (2 секунды)

```bash
python src/synth/generator.py 10000 runs/drafts_clean_10k.jsonl 0
```

Последний аргумент `0` — без ASR-шума. Учитель получает чистый текст, шум добавляется после.
Генератор детерминированный: у всех получится один и тот же файл.

### 3.3. Пробный прогон на 20 диалогах — обязательно

```bash
python src/synth/run_distill.py --n 20 --workers 4 --out runs/distill_probe.jsonl
```

В конце смотри строку `готово: {'ok': …, 'rejected': …, 'error': …}`.

- **ok ≥ 14 из 20** — всё хорошо, иди дальше.
- **Много `error`.** Причины смотри в `runs/distill_probe_rejects.jsonl`:
  - `NotFoundError` — неверный `TEACHER_MODEL` или адрес;
  - `BadRequestError` про `response_format` — сервер не умеет JSON-режим, добавь флаг `--no_json_mode`;
  - таймауты — уменьши `--workers`.
- **Много `rejected`.** Модель теряет факты или даты. Попробуй модель крупнее или `--temperature 0.5`.

Глазами проверь 2–3 диалога из `runs/distill_probe.jsonl`. Реплики должны быть живыми, с «ну», «эээ»,
перебивками. Факты из поля `facts` должны реально звучать в реплике `source.turn_idx`, даты — словами
(«со вторника», «позавчера»).

### 3.4. Основной прогон — 10 000

```bash
python src/synth/run_distill.py --n 10000 --workers $WORKERS --out runs/distilled_10k.jsonl
```

- Прогресс печатается каждые 25 диалогов, с оценкой оставшегося времени.
- **Можно прерывать** (Ctrl+C, сон ноутбука, падение сервера). Повторный запуск той же командой
  продолжит с места остановки: уже обработанные `call_id` пропускаются.
- Хорошие диалоги пишутся в `runs/distilled_10k.jsonl`, отбракованные — в `runs/distilled_10k_rejects.jsonl`.
- Ожидаемо 70–90% ok. Чтобы получить больше хороших, сгенерируй, например, 13 000 черновиков и запусти с `--n 13000`.
- Полезные флаги:
  - `--noise 0.25` — уровень ASR-шума поверх ответа (по умолчанию 0.25, как в реальных звонках);
  - `--temperature 0.7`;
  - `--max_tokens 3000`.

Что проверяет код в каждом ответе (`teacher.validate`):
1. Значение факта действительно звучит в реплике-источнике или соседней.
2. Хотя бы одна граница даты восстанавливается из текста (±1 день).
3. Если прошло меньше 80% фактов плана, весь диалог бракуется.

Проверка протестирована: правильные ответы проходят в 95% случаев, мусор — в 0%.

### 3.5. Дообучить модель и сравнить

```bash
# 1) экстрактор с дистиллированными данными (они идут только в fit-часть; ранжировщик калибруется на реальных)
python src/fit_v4.py --extra runs/distilled_10k.jsonl --out runs/v4_distill.pkl
# 2) селектор текущего значения
BASE_GENES=runs/evo2_best.json python src/run_current.py runs/v4_distill.pkl runs/v5_distill.pkl
# 3) метрики: было и стало
python src/eval_val.py --model models/compactfact_v5.pkl --genes models/genes_best.json --tag before
python src/eval_val.py --model runs/v5_distill.pkl        --genes models/genes_best.json --tag distill
# 4) (если стало лучше на golden/dev_B) переподобрать гены под новую модель
python src/run_evolve.py --model runs/v5_distill.pkl --tag evo5 --base models/genes_best.json --workers 4
```

**Как читать результат.**
- Решение «стало лучше или нет» принимай по **golden и dev_B**, а не по val. val — финальная проверка;
  если подбирать по нему, мы переобучимся под тест.
- Шум метрики на 150 звонках — σ ≈ 0.04, на 383 — ≈ 0.025. Прирост меньше ~0.02 — это шум.
- Если 10 000 дают меньше, чем хотелось, попробуй `--extra_n 3000`: иногда меньше синтетики лучше.
- Если прирост есть, проверь, что модель по-прежнему ≤ 1 с на звонок:

```bash
python src/benchmark.py --model runs/v5_distill.pkl --genes models/genes_best.json --out runs/bench_distill.json
```

### 3.6. Что прислать обратно

Закоммить в ветку `compactfact-v2`:
- `runs/distill_*_eval.json`, `runs/before_eval.json`;
- `runs/evo5_best.json` и `runs/evo5.log`, если запускал эволюцию;
- `runs/bench_distill.json`.

Сам `runs/distilled_10k.jsonl` (≈ 35–40 МБ) можно закоммитить или передать архивом.

`.pkl`-модель (≈ 17–40 МБ) клади в `models/compactfact_v6.pkl`, только если она лучше на golden и dev_B.
Если пушить файл больше 100 МБ, GitHub его не примет — передай отдельно.

---

## 4. Если что-то пошло не так

| симптом | что делать |
|---|---|
| `could not import langchain_openai` | `pip install -r deploy/requirements.txt` |
| запросы идут, но `ok=0` | открой 1–2 строки rejects и сам ответ модели: `python -c "import sys;sys.path+=['src','src/synth'];import teacher,json;c=json.loads(open('runs/drafts_clean_10k.jsonl').readline());print(teacher.distill_chain().invoke({'style':'{}','started_at':c['started_at'],'plan':json.dumps(teacher.plan_of(c),ensure_ascii=False),'draft':'\n'.join(f\"{t['idx']} {t['speaker']}: {t['text']}\" for t in c['turns'])}))"` |
| модель «думает» вслух, очень медленно | убедись, что `TEACHER_NO_THINK` не равен 0; для vLLM возьми `-Instruct-2507`-версию |
| OOM в vLLM | `--max-model-len 6144`, `--gpu-memory-utilization 0.85` или AWQ-версия |
| fit_v4 долго / не хватает RAM | `--extra_n 5000`; обучение одно-поточное (OMP_NUM_THREADS=1 выставлен специально) |
| хочется через DeepSeek API вместо локальной | `TEACHER_BASE_URL=https://api.deepseek.com TEACHER_MODEL=deepseek-chat TEACHER_API_KEY=sk-…`; ключ не коммитить (`.env` в `.gitignore`) |

Вопросы по коду — смотри докстринги в `src/synth/teacher.py` и `src/synth/run_distill.py`.
