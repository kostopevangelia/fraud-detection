FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System deps (αν χρειαστούν για numpy/pandas/scikit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Μόνο ό,τι χρειάζεται στο runtime
COPY app/ app/
COPY models/ models/

# Προαιρετικά: περιβάλλον για paths
ENV MODEL_DIR=/app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]