# ==== CONFIG ====
# Python executable (override with: make PY=python3.11)
PY ?= python

# Directories
MODEL_DIR ?= models
TRAIN_DIR ?= training
RES_DIR ?= resources

# FastAPI app module
APP_MODULE ?= app.main:app
UVICORN_PORT ?= 8000

# ==== PATHS ====
GEN_PY   := $(TRAIN_DIR)/generate_data.py
TRAIN_PY := $(TRAIN_DIR)/train_custom.py
DATA_CSV := $(RES_DIR)/custom_transactions.csv
MODEL_PKL := $(MODEL_DIR)/fraud_model_custom.pkl
FEATS_PKL := $(MODEL_DIR)/custom_model_features.pkl

# ==== PHONY ====
.PHONY: all retrain generate train api clean docker-build docker-up docker-restart check

# Default: full retrain
all: retrain

# Step 1: Generate synthetic data (writes to resources/custom_transactions.csv)
generate:
	$(PY) $(GEN_PY)

# Step 2: Train model and save artifacts into $(MODEL_DIR)
# NOTE: train_custom.py should accept --outdir and --data
train:
	$(PY) $(TRAIN_PY) --outdir $(MODEL_DIR) --data $(DATA_CSV)

# Full pipeline
retrain: generate train
	@echo "✅ Retrain completed. Artifacts:"
	@ls -l $(MODEL_PKL) $(FEATS_PKL) 2>/dev/null || dir $(MODEL_DIR)

# Run API locally (no inline env vars → works on Windows & Unix)
api:
	uvicorn $(APP_MODULE) --host 0.0.0.0 --port $(UVICORN_PORT) --reload

# Clean model artifacts (⚠ use with caution)
clean:
	@rm -f $(MODEL_PKL) $(FEATS_PKL) 2>/dev/null || del /Q $(MODEL_DIR)\fraud_model_custom.pkl $(MODEL_DIR)\custom_model_features.pkl
	@echo "🧹 Removed model artifacts."

# ---- Docker helpers ----
docker-build:
	docker build -t fraud-api:latest .

docker-up:
	docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models --name fraud-api fraud-api:latest

docker-restart:
	- docker stop fraud-api
	docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models --name fraud-api fraud-api:latest

# Send a demo request to the running API (works without jq)
check:
	@echo "Sending test transaction to http://localhost:8000/predict-fraud ..."
	@curl -s -X POST "http://localhost:8000/predict-fraud" -H "Content-Type: application/json" -d "{\"amount\":3500.0,\"currency\":\"USD\",\"paymentType\":\"card\",\"transactionType\":\"PAYMENT\",\"userId\":\"USER123\",\"bin\":\"400005\",\"hour\":2,\"day_of_week\":6}"
	@echo
