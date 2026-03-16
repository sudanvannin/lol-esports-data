FROM python:3.11-slim

WORKDIR /app

# Install dependencies required by the Web App
COPY requirements.txt ./
# We only need the web/duckdb dependencies for the frontend
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    jinja2 \
    duckdb \
    pandas

# Copy application code
COPY web/ ./web/
COPY src/ ./src/

# Set env variables
ENV PORT=10000
ENV PYTHONPATH=/app

# Expose port (Render automatically uses $PORT, default is 10000)
EXPOSE 10000

# Run uvicorn server
CMD uvicorn web.app:app --host 0.0.0.0 --port $PORT
