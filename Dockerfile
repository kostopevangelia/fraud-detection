FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]