from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .ingestion import CorpusIndex
from .models import RetrievedChunk, Ticket


def _normalize_company(value: str) -> str:
    lowered = (value or "").strip().lower()
    if "visa" in lowered:
        return "visa"
    if "hacker" in lowered:
        return "hackerrank"
    if "anthropic" in lowered or "claude" in lowered:
        return "claude"
    return "general"


def _simple_query_rewrite(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.replace("acnt", "account").replace("pwd", "password").replace("txn", "transaction")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class Retriever:
    index: CorpusIndex
    top_k: int
    rrf_k: int

    def __post_init__(self) -> None:
        self.bm25 = BM25Okapi(self.index.bm25_corpus_tokens)

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        query = _simple_query_rewrite(ticket.text())
        query_emb = self.index.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        sem_scores, sem_indices = self.index.faiss_index.search(query_emb, min(len(self.index.chunks), max(self.top_k * 5, 10)))
        sem_scores = sem_scores[0].tolist()
        sem_indices = sem_indices[0].tolist()

        bm25_scores = self.bm25.get_scores(query.split())
        ranked_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: max(self.top_k * 5, 10)]

        sem_rank = {idx: rank for rank, idx in enumerate(sem_indices, start=1)}
        bm25_rank = {idx: rank for rank, idx in enumerate(ranked_bm25_idx, start=1)}
        all_candidates = set(sem_rank) | set(bm25_rank)

        company_bias = _normalize_company(ticket.company)

        fused: list[tuple[int, float]] = []
        for idx in all_candidates:
            score = 0.0
            if idx in sem_rank:
                score += 1.0 / (self.rrf_k + sem_rank[idx])
            if idx in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[idx])

            chunk_company = self.index.chunks[idx].company.lower()
            if company_bias != "general" and company_bias in chunk_company:
                score *= 1.25
            elif company_bias != "general":
                score *= 0.9
            fused.append((idx, score))

        fused_sorted = sorted(fused, key=lambda x: x[1], reverse=True)[: self.top_k]

        results: list[RetrievedChunk] = []
        sem_score_map = {idx: sem_scores[rank - 1] for idx, rank in sem_rank.items() if rank - 1 < len(sem_scores)}
        for rank, (idx, _) in enumerate(fused_sorted, start=1):
            chunk = self.index.chunks[idx].model_copy()
            chunk.semantic_score = float(sem_score_map.get(idx, 0.0))
            chunk.bm25_score = float(bm25_scores[idx] if idx < len(bm25_scores) else 0.0)
            chunk.fused_rank = rank
            results.append(chunk)

        return results

    @staticmethod
    def confidence(chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        return sum(max(c.semantic_score, 0.0) for c in chunks) / max(len(chunks), 1)
