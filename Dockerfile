# Dockerfile (FastAPI + Playwright)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by Playwright browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates wget unzip \
    libnss3 libatk-bridge2.0-0 libx11-6 libxcomposite1 libxrandr2 \
    libxdamage1 libxfixes3 libgbm1 libpangocairo-1.0-0 libcups2 \
    libxcb-dri3-0 libxkbcommon0 libasound2 libatk1.0-0 libgtk-3-0 \
    fonts-liberation fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Playwright browser binaries (with OS deps)
# Use python -m playwright to be explicit
RUN python -m playwright install --with-deps

# Copy app source
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

