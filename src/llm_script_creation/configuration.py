"""Project configuration.

Values come from the project `.env` file where possible. Anything that is
specific to one of the two Uwazi instances lives in `db.py` as a constant.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# --- LLM -------------------------------------------------------------------
# `MODEL` is intentionally a plain module-level constant: it is the single
# place to change the model used to synthesise extraction scripts.
MODEL = os.getenv("MODEL", "deepseek-v4.1-flash:cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
OLLAMA_NATIVE_BASE = OLLAMA_BASE_URL[:-3] if OLLAMA_BASE_URL.endswith("/v1") else OLLAMA_BASE_URL
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# --- Storage ---------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
WORK_DIR = Path(os.getenv("WORK_DIR", PROJECT_ROOT / "work"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
CATALOG_PATH = ARTIFACTS_DIR / "catalog.db"
STATES_PATH = ARTIFACTS_DIR / "states.json"

for _directory in (REPORTS_DIR, RESULTS_DIR, WORK_DIR, ARTIFACTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
