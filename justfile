start:
    docker compose up -d

stop:
    docker compose stop

down:
    docker compose down

sync:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv sync --group dev

inspect:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv run inspect-data

build:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv run build-datasets

app:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv run streamlit run app/app.py

test:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv run pytest -q

lint:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv run ruff check src tests

format:
    UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache} uv run ruff format src tests
