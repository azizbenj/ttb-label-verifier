FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Local OCR engine + English model (~30 MB). No network access is needed at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Runtime files only: the sample labels feed the "Try a sample" menu and the sample batch.
COPY app ./app
COPY data/samples ./data/samples
COPY data/batch ./data/batch

# Nothing is written at runtime except the web framework's temporary upload files in /tmp.
RUN useradd --system --uid 10001 --no-create-home labelcheck
USER labelcheck

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8000'), timeout=4)"
# One process on purpose: batch jobs live in its memory. exec makes uvicorn PID 1 so it receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
