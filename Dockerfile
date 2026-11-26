# 1. Base image: lightweight Python 3.12 (as specified in handbook)
FROM python:3.12-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements and install dependencies BEFORE copying code (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy required code and model directories
COPY app/ app/
COPY src/ src/
COPY models/ models/

# 5. Set PYTHONPATH so Python finds packages inside src/
ENV PYTHONPATH=/app/src

# 6. Expose API port
EXPOSE 8000

# 7. Start Uvicorn web server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
