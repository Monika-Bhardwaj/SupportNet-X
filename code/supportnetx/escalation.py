from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import RiskResult


CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

RISK_KEYWORDS = {
    "fraud",
    "unauthorized",
    "hacked",
    "security breach",
    "social security",
    "medical advice",
    "legal advice",
    "wire transfer",
    "credit card",
    "bank account",
    "password leak",
    "exploit",
    "bypass",
}


@dataclass
class EscalationLogic:
    min_confidence: float

    def evaluate_text_risk(self, text: str) -> RiskResult:
        lowered = text.lower()
        for keyword in RISK_KEYWORDS:
            if keyword in lowered:
                return RiskResult(
                    escalated=True,
                    reason=f"Escalated: sensitive topic detected ({keyword}).",
                    category="security_risk" if keyword in {"hacked", "unauthorized", "security breach", "exploit", "bypass"} else "financial_risk",
                )
        if CARD_RE.search(text):
            return RiskResult(
                escalated=True,
                reason="Escalated: possible payment card number detected (PCI-sensitive).",
                category="pii_detected",
            )
        if EMAIL_RE.search(text) and "password" in lowered:
            return RiskResult(
                escalated=True,
                reason="Escalated: credentials/PII-like content detected.",
                category="pii_detected",
            )
        return RiskResult(escalated=False, reason="No direct risk keyword or PII pattern detected.", category="none")

    def evaluate_retrieval_confidence(self, similarities: Iterable[float]) -> RiskResult:
        values = list(similarities)
        if not values:
            return RiskResult(
                escalated=True,
                reason="Escalated: no relevant support documents retrieved.",
                category="out_of_scope",
            )
        avg_score = sum(values) / len(values)
        if avg_score < self.min_confidence:
            return RiskResult(
                escalated=True,
                reason=f"Escalated: low retrieval confidence (avg={avg_score:.3f} < {self.min_confidence:.3f}).",
                category="low_confidence",
            )
        return RiskResult(
            escalated=False,
            reason=f"Retrieval confidence acceptable (avg={avg_score:.3f}).",
            category="none",
        )
