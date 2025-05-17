FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY scripts/ ./scripts/
RUN mkdir -p data

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY src/ ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Debug step (optional)
RUN ls -1 /app/.venv/bin

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
