# ─────────────────────────────────────────────────────────────
# Dockerfile — Supply Chain Ghost Protocol
# Optimized for: 2 vCPUs / 8GB RAM (hackathon infra constraint)
# Target: Hugging Face Spaces (Docker SDK)
# ─────────────────────────────────────────────────────────────

# Stage 1: Dependency builder (separate layer for cache efficiency)
FROM python:3.11-slim AS builder

# Build deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first (cache layer)
COPY requirements.txt .

# Install into isolated prefix — avoids polluting system Python
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime image (minimal)
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime OS deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 appuser
USER appuser

WORKDIR /app

# Copy project source
COPY --chown=appuser:appuser . .

# ─────────────────────────────────────────────────────────────
# Environment configuration
# Actual secrets injected at runtime via HF Space secrets
# ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # CPU threading — tuned for 2 vCPU constraint
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    # Pydantic v2 performance
    PYDANTIC_V2_STRICT=1

# Port for FastAPI server (HF Spaces default)
EXPOSE 7860

# ─────────────────────────────────────────────────────────────
# Health check — validates OpenEnv API is responsive
# ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "app_server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "asyncio", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
