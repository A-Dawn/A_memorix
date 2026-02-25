FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY amemorix ./amemorix
COPY core ./core
COPY scripts ./scripts
COPY web ./web
COPY server.py __init__.py config.toml.example ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install .

EXPOSE 8082
VOLUME ["/app/data"]

CMD ["python", "-m", "amemorix", "serve"]

