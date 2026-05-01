from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]
TicketStatus = Literal["replied", "escalated"]


class Ticket(BaseModel):
    issue: str = ""
    subject: str = ""
    company: str = ""

    def text(self) -> str:
        return f"Subject: {self.subject}\nIssue: {self.issue}".strip()


class RetrievedChunk(BaseModel):
    chunk_id: str
    company: str
    source_path: str
    text: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    fused_rank: int = 9999


class ClassificationResult(BaseModel):
    request_type: RequestType
    product_area: str
    justification: str = Field(min_length=1)


class RiskResult(BaseModel):
    escalated: bool
    reason: str


class ResponsePayload(BaseModel):
    response: str
    justification: str


class TicketOutput(BaseModel):
    status: TicketStatus
    product_area: str
    response: str
    justification: str
    request_type: RequestType
