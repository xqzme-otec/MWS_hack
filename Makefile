# Короткие команды. Docker: up / eval / train / smoke.  Без Docker: install / serve / evaluate.
.PHONY: up eval train train-gpu smoke install serve evaluate report demo

up:            ## веб-интерфейс + API на http://localhost:8000
	docker compose up --build
eval:          ## метрики val + adversarial, разбор ошибок, замер ресурсов, графики
	docker compose --profile eval run --rm eval
train:         ## обучение с нуля на CPU (долго; см. train-gpu)
	docker compose --profile train run --rm train
train-gpu:     ## обучение с нуля на NVIDIA GPU
	docker compose --profile train-gpu run --rm train-gpu
smoke:         ## смоук-тест обучения (≈10 мин CPU)
	docker compose --profile train run --rm -e SMOKE=1 train

install:
	pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
serve:
	uvicorn --app-dir src api:app --host 0.0.0.0 --port 8000
evaluate:
	python src/evaluate.py --sets val,val_adv && python src/error_report.py && python src/bench_resources.py && python src/make_report.py
report:
	python src/make_report.py
demo:
	python src/demo_gliner.py --i 5
