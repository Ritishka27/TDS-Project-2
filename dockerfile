FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps: include poppler-utils for PDF support + Playwright deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates wget unzip \
    libnss3 libatk-bridge2.0-0 libx11-6 libxcomposite1 libxrandr2 \
    libxdamage1 libxfixes3 libgbm1 libpangocairo-1.0-0 libcups2 \
    libxcb-dri3-0 libxkbcommon0 libasound2 libatk1.0-0 libgtk-3-0 \
    fonts-liberation fonts-dejavu-core poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# If you use Playwright, install its browsers in build
RUN python -m playwright install --with-deps

# Copy app
COPY . /app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]



