from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import RetrievedChunk


def _chunk_text(text: str, target_chars: int = 450) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = []
    current_len = 0
    for paragraph in paragraphs:
        if current_len + len(paragraph) > target_chars and current:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _list_doc_files(data_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv"}
            and "index_cache" not in path.parts
        ]
    )


def _files_fingerprint(paths: Iterable[Path]) -> str:
    payload = []
    for path in paths:
        stat = path.stat()
        payload.append(f"{path}|{int(stat.st_mtime)}|{stat.st_size}")
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


@dataclass
class CorpusIndex:
    chunks: list[RetrievedChunk]
    embeddings: np.ndarray
    faiss_index: faiss.IndexFlatIP
    model: SentenceTransformer
    bm25_corpus_tokens: list[list[str]]


def build_or_load_index(data_dir: Path, cache_dir: Path, embedding_model: str) -> CorpusIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _list_doc_files(data_dir)
    fingerprint = _files_fingerprint(files) if files else "empty"
    meta_path = cache_dir / "meta.json"
    payload_path = cache_dir / "payload.pkl"

    if meta_path.exists() and payload_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint and meta.get("embedding_model") == embedding_model:
            with payload_path.open("rb") as handle:
                payload = pickle.load(handle)
            model = SentenceTransformer(embedding_model)
            index = faiss.IndexFlatIP(payload["embeddings"].shape[1])
            index.add(payload["embeddings"].astype("float32"))
            return CorpusIndex(
                chunks=payload["chunks"],
                embeddings=payload["embeddings"],
                faiss_index=index,
                model=model,
                bm25_corpus_tokens=payload["bm25_tokens"],
            )

    chunks: list[RetrievedChunk] = []
    for file_path in files:
        company = file_path.parts[file_path.parts.index("data") + 1] if "data" in file_path.parts[:-1] else "general"
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for idx, chunk in enumerate(_chunk_text(text)):
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{file_path.stem}-{idx}",
                    company=company.lower(),
                    source_path=str(file_path),
                    text=chunk,
                )
            )

    model = SentenceTransformer(embedding_model)
    texts = [chunk.text for chunk in chunks] or [""]
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    if not chunks:
        chunks = [
            RetrievedChunk(
                chunk_id="empty-0",
                company="general",
                source_path="",
                text="No documents loaded.",
            )
        ]
    bm25_tokens = [chunk.text.lower().split() for chunk in chunks]

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    with payload_path.open("wb") as handle:
        pickle.dump({"chunks": chunks, "embeddings": emb, "bm25_tokens": bm25_tokens}, handle)
    meta_path.write_text(
        json.dumps({"fingerprint": fingerprint, "embedding_model": embedding_model}, indent=2),
        encoding="utf-8",
    )

    return CorpusIndex(
        chunks=chunks,
        embeddings=emb,
        faiss_index=index,
        model=model,
        bm25_corpus_tokens=bm25_tokens,
    )
