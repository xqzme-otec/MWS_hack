# CompactFact v2 — компактный экстрактор фактов (кейс МТС MWS CompactFactExtraction)

Извлекает структурированные факты (состояние услуги, тариф, статус заявки, действие клиента,
договорённость) из зашумлённых русскоязычных транскриптов звонков в техподдержку. Работает на
CPU: p50 157 мс / p95 362 мс на звонок в один поток, без GPU. Оценка — официальным `case_metrics.py` v2.1.

## Результат на val (официальный оценщик v2.1)

| метрика | база glm-5.3 | CompactFact v2 (v5 + эволюция 4) | критерий |
|---|---|---|---|
| temporal_recall | 0.4706 | **0.5344** | ≥ 0.4506 |
| state_accuracy | 0.5345 | **0.5746** (строгая без утечки: 0.5730) | ≥ 0.5145 |
| fact_precision / recall | 0.3348 / 0.3805 | 0.5131 / 0.4141 | — |
| grounding_turn | 0.7835 | 0.7997 | — |
| val + ASR-шум 0.3: T / S | — | 0.5242 / 0.5490 | — |
| корректность схемы | — | 383/383 звонков | 100% |
| латентность / память | — | 157 мс p50, 344 МБ, 0 GPU-с | ~1 с |

Подробности, кривая «ресурсы × качество», разбор ошибок и найденные баги оценщика — в отчёте.

## Архитектура

```
транскрипт → [кандидаты: char-TF-IDF прототипы значений × реплики]
           → [ранжировщик HistGBM на атрибут: сходства, покрытие слов, ловушки, говорящий, контекст]
           → (опционально) [GLiNER-x-small + LoRA: спаны фактов → усиление/новые кандидаты;
              эволюция 4 его выключила: +0.004 к T ценой ×20 латентности на CPU]
           → [фильтр ловушек схемы §4] → [объект: каталог + обучаемый выбор]
           → [даты: банк относительных выражений + обучаемый выбор кандидата]
           → [текущее значение: обучаемый селектор, одно открытое значение на объект]
           → факты в схеме кейса (+ валидация схемы)
```

Параметры сборки (пороги, режимы ловушек/дат/цепочек, параметры движка GLiNER) подбираются
эволюцией (`src/cfx/evolve.py`) по golden-набору (150 звонков) с перепроверкой на отложенном dev.

## Быстрый старт

```bash
pip install -r deploy/requirements.txt
python src/fit_v4.py                                                   # обучение с cross-fitting (≈8 мин, CPU)
BASE_GENES=runs/evo2_best.json python src/run_current.py runs/v4.pkl runs/v5.pkl   # селектор текущего значения
python src/run_evolve.py --model runs/v5.pkl --tag evo3 --base runs/evo2_best.json
python src/run_evolve_gliner.py                                        # эволюция 4 (гены движка GLiNER)
python data/case_metrics.py data/val.jsonl runs/evo4_val.jsonl          # официальная оценка
CFX_MODEL=models/compactfact_v5.pkl CFX_GENES=models/genes_best.json uvicorn src.api:app --port 8000
```

`POST /extract` принимает `{call_id, started_at, turns:[{idx,speaker,text}]}` и возвращает звонок с `facts`.
Docker: `docker compose -f deploy/compose.yaml up --build`.

## Состав

- `src/cfx/` — модель: `extractor.py` (кандидаты+ранжирование+сборка), `dates.py`/`dates_bank.py` (даты),
  `objects.py` (объекты), `current.py` (текущее значение), `gliner_fusion.py`, `gliner_data.py`, `denoise.py`,
  `evolve.py` (эволюция), `bench.py` (стенд), `schema.py` (валидация).
- `src/synth/` — `markers.py` (маркеры речи по open_stt и корпусу), `generator.py` (синтетика по построению),
  `teacher.py` (LangChain: учитель — любой OpenAI-совместимый сервер: vLLM/Ollama/DeepSeek; верификатор спорных фактов,
  экстрактор как Runnable), `run_distill.py` (массовая дистилляция с параллелизмом и продолжением).
- `src/eval_val.py` — метрики golden / dev_B / val одной командой.
- **`HANDOFF.md` — инструкция по дистилляции 10 000 диалогов через локальную LLM.**
- `src/train_gliner.py` — LoRA-дообучение GLiNER; `src/benchmark.py` — замеры ресурсов; `src/robust.py` — шумовой стресс.
- `src/scorer_fixed.py`, `src/scorer_strict.py` — копии оценщика с исправленными багами (только для анализа).
- `runs/` — логи эволюции, метрики, предсказания; `models/` — веса.

Сиды фиксированы (42) во всех обучениях; разбиение train по аккаунтам (md5): fit 70% / rank 15% / dev 15%.
val не используется ни для обучения, ни для подбора генов — только для итоговой оценки.

Подробный отчёт: Docs-артефакт «CompactFact v2 — отчёт по переделке решения» (ссылка в чате).
Данные кейса v2.1 уже лежат в `data/`. Адаптер GLiNER (105 МБ) в git не входит (лимит GitHub, для лучшей конфигурации не нужен).
