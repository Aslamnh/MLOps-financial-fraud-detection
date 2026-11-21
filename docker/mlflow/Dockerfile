FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Versi disamakan dengan yang dipakai saat melatih dan mendaftarkan model.
RUN pip install mlflow==3.17.1

EXPOSE 5000
