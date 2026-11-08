FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/usr/local UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
