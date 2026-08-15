# Container image for the forecasting API.
#
# Why Docker (in one sentence): it freezes the exact environment — Python
# version, library versions, code — into one image, so the model that passed
# evaluation on your laptop is byte-for-byte the model that serves on EC2.
#
# Build & run:
#   docker build -t demand-forecast-api .
#   docker run -p 8000:8000 demand-forecast-api
#
# NOTE: expects a trained model in artifacts/ and data in data/ — run
#   python run_pipeline.py  before building.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, in their own layer: Docker caches layers, so
# code changes don't re-download PyTorch every build.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/
COPY data/demand.csv data/demand.csv

EXPOSE 8000

# --workers 1 : the model lives in memory; one worker per (small) instance.
# Horizontal scaling = more EC2 nodes behind a load balancer, not more workers.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
