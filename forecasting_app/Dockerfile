# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make cmake \
    curl gnupg unixodbc unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: Microsoft ODBC Driver 18 for SQL Server (comment out if you have issues)
RUN set -eux; \
    arch=$(dpkg --print-architecture); \
    if [ "$arch" = "amd64" ] || [ "$arch" = "arm64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends apt-transport-https ca-certificates && \
      curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
      curl -sSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/microsoft-prod.list && \
      apt-get update && apt-get install -y --no-install-recommends msodbcsql18 && \
      rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]