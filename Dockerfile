# One image with the API, the web page and the released model in model/.

# --- 1. Install ---
FROM python:3.13-slim AS build
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.13.10
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# --- 2. What runs ---
FROM python:3.13-slim
RUN useradd --uid 10001 --create-home eta
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY model ./model
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
USER eta
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD \
    ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/ready', timeout=4)"]
# `exec` lets the host's stop signal reach the server, so it finishes open requests.
CMD ["sh", "-c", "exec python -m delivery_eta serve --host 0.0.0.0 --port ${PORT}"]
