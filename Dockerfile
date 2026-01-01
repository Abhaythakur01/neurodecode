# Multi-stage Dockerfile for NeuroDecode backend

# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: Development image
FROM dependencies as development

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run development server
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production image
FROM dependencies as production

# Create non-root user
RUN useradd -m -u 1000 neurodecode && \
    mkdir -p /app /data /models && \
    chown -R neurodecode:neurodecode /app /data /models

# Switch to non-root user
USER neurodecode

WORKDIR /app

# Copy only necessary files
COPY --chown=neurodecode:neurodecode setup.py pyproject.toml ./
COPY --chown=neurodecode:neurodecode src/ ./src/

# Install package
RUN pip install --user .

# Set PATH for user-installed packages
ENV PATH="/home/neurodecode/.local/bin:${PATH}"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run production server
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
