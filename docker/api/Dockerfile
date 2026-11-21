FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY docker/api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY api/ ./api/

RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
