# 1. Use the official lightweight Python base image
FROM python:3.11.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency file first (better Docker cache usage)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy project files
COPY . .

# 6. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Run FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]