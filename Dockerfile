# One image with the API, the web page and a freshly trained model.
# The model is built here from the code and data, so an image can't hold a model that
# doesn't match its code. If the model fails the quality gate, the build fails.

# --- 1. Install, train and check ---
FROM python:3.13-slim AS build
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
# LightGBM needs this library, which the slim image leaves out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.10
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra train --no-install-project
COPY src ./src
COPY configs ./configs
COPY data/raw/train.csv ./data/raw/train.csv
RUN uv sync --frozen --no-dev --extra train --no-editable
RUN .venv/bin/python -m delivery_eta train
# Leave the training libraries behind.
RUN uv sync --frozen --no-dev --no-editable

# --- 2. What runs ---
FROM python:3.13-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home eta
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY --from=build /app/artifacts ./artifacts
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
USER eta
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD \
    ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/ready', timeout=4)"]
# `exec` lets the host's stop signal reach the server, so it finishes open requests.
CMD ["sh", "-c", "exec python -m delivery_eta serve --host 0.0.0.0 --port ${PORT}"]
