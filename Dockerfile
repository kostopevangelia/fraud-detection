FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements if you have them
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and model files
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "src.fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]