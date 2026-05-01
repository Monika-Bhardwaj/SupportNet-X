from __future__ import annotations

from dataclasses import dataclass

from .models import ClassificationResult, RequestType, Ticket


PRODUCT_AREA_KEYWORDS = {
    "account/billing": ["billing", "invoice", "payment", "subscription", "refund", "charge"],
    "fraud/security": ["fraud", "hacked", "unauthorized", "security", "phishing", "otp"],
    "technical": ["error", "bug", "crash", "timeout", "api", "integration", "latency"],
    "assessments": ["assessment", "test case", "compiler", "candidate", "score", "submission"],
}


@dataclass
class Classifier:
    def classify(self, ticket: Ticket) -> ClassificationResult:
        text = ticket.text().lower()

        request_type: RequestType = "product_issue"
        if any(token in text for token in ["feature", "enhancement", "would like", "add support"]):
            request_type = "feature_request"
        elif any(token in text for token in ["bug", "broken", "error", "fails", "not working", "crash"]):
            request_type = "bug"
        elif any(token in text for token in ["hello", "test", "random", "n/a", "na", "dummy"]):
            request_type = "invalid"

        best_area = "general"
        best_score = -1
        for area, keywords in PRODUCT_AREA_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_area = area
                best_score = score

        if best_score == 0:
            best_area = "technical"

        return ClassificationResult(
            request_type=request_type,
            product_area=best_area,
            justification=f"Classified from ticket keywords; request_type={request_type}, product_area={best_area}.",
        )
