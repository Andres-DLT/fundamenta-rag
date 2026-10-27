# Imagen de producción de Fundamenta (CPU).
#
#   docker build -t fundamenta .
#   docker run -p 7860:7860 -e FUNDAMENTA_PASSWORD=... fundamenta
#
# Configuración mediante variables de entorno (ver docs/despliegue.md).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/fundamenta

# PyTorch para CPU primero: es la dependencia más pesada y cambia poco
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY fundamenta ./fundamenta
RUN pip install ".[semantic,claude]"

COPY --chown=app data ./data

USER app

# El modelo de embeddings se descarga durante la construcción para que el arranque sea inmediato
RUN python -c "from fundamenta.embeddings import SentenceTransformerEmbedder; SentenceTransformerEmbedder()"

ENV PORT=7860 \
    FUNDAMENTA_DATA=data/novatech \
    FUNDAMENTA_EMBEDDER=semantic \
    FUNDAMENTA_LLM=none

EXPOSE 7860

CMD ["sh", "-c", "exec fundamenta serve --host 0.0.0.0 --port \"$PORT\""]
