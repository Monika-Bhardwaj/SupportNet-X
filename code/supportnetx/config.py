from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    data_dir: Path
    tickets_csv: Path
    output_csv: Path
    cache_dir: Path
    embedding_model: str
    top_k: int
    min_confidence: float
    rrf_k: int
    max_concurrency: int
    seed: int
    llm_provider: str
    llm_model: str
    query_rewrite_model: str
    enable_query_rewrite: bool
    strict_company_routing: bool
    log_file: Path


def _default_log_file() -> Path:
    home = Path.home()
    return home / "hackerrank_orchestrate" / "log.txt"


def load_config(project_root: Path) -> Config:
    data_dir = project_root / "data"
    tickets_csv = project_root / "support_tickets" / "support_tickets.csv"
    output_csv = project_root / "support_tickets" / "output.csv"
    cache_dir = data_dir / "index_cache"
    log_file = Path(os.getenv("AGENTS_LOG_FILE", str(_default_log_file())))

    return Config(
        data_dir=data_dir,
        tickets_csv=tickets_csv,
        output_csv=output_csv,
        cache_dir=cache_dir,
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=int(os.getenv("TOP_K", "3")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.25")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "8")),
        seed=int(os.getenv("SUPPORTNETX_SEED", "42")),
        llm_provider=os.getenv("LLM_PROVIDER", "huggingface"),
        llm_model=os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
        query_rewrite_model=os.getenv("QUERY_REWRITE_MODEL", "gpt-4o-mini"),
        enable_query_rewrite=os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true",
        strict_company_routing=os.getenv("STRICT_COMPANY_ROUTING", "false").lower() == "true",
        log_file=log_file,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
