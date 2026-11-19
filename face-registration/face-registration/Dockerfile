FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Debian Trixie compatible)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    python3-dev \
  && pip install --upgrade pip setuptools wheel \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sed -i 's/numpy==1.24.3/numpy==1.26.0/' requirements.txt || true
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 1010

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "1010", "--reload"]
