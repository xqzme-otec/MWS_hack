# CompactFact — GLiNER g13-ru + бустинг, CPU-only. Одна команда:  docker compose up --build
FROM python:3.13-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 CFX_THREADS=2 \
    HF_HOME=/app/.hf TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY web/ web/
COPY configs/ configs/
COPY scripts/ scripts/
COPY examples/ examples/
COPY models/ models/
COPY data/catalog.json data/case_metrics.py data/schema.md data/
# веса GLiNER лежат в репозитории частями < 50 МБ — склеиваем при сборке
RUN python -c "import sys; sys.path.insert(0, 'src'); from cfx.pipeline import ensure_weights; ensure_weights('models/g13_ru')" \
 && rm -f models/g13_ru/pytorch_model.bin.part* \
 && mkdir -p reports runs .hf
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "--app-dir", "src", "api:app", "--host", "0.0.0.0", "--port", "8000"]
