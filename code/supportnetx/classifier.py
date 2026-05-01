from __future__ import annotations

from dataclasses import dataclass

from .models import ClassificationResult, RequestType, Ticket


PRODUCT_AREA_KEYWORDS = {
    "account/billing": ["billing", "invoice", "payment", "subscription", "refund", "charge", "visa card", "transaction", "pricing", "credit card"],
    "fraud/security": ["fraud", "hacked", "unauthorized", "security", "phishing", "otp", "stolen", "compromised", "verify identity"],
    "technical": ["error", "bug", "crash", "timeout", "api", "integration", "latency", "broken", "fails", "not working", "compiler", "runtime"],
    "assessments": ["assessment", "test case", "compiler", "candidate", "score", "submission", "interview", "hackerank test", "contest"],
    "account/access": ["login", "password", "reset", "access", "account", "locked", "signin", "sso", "authentication"],
    "permissions": ["permission", "access level", "admin", "role", "authorize", "grant", "denied", "restricted"],
}


@dataclass
class Classifier:
    def classify(self, ticket: Ticket) -> ClassificationResult:
        text = ticket.text().lower()

        request_type: RequestType = "product_issue"
        if any(token in text for token in ["feature", "enhancement", "would like", "add support", "improvement", "request for"]):
            request_type = "feature_request"
        elif any(token in text for token in ["bug", "broken", "error", "fails", "not working", "crash", "incorrect behavior", "wrong result"]):
            request_type = "bug"
        elif any(token in text for token in ["hello", "test", "random", "n/a", "na", "dummy", "adsf", "qwerty"]):
            request_type = "invalid"

        best_area = "general"
        best_score = -1
        
        # Calculate scores for each area
        scores = {}
        for area, keywords in PRODUCT_AREA_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[area] = score
            if score > best_score:
                best_area = area
                best_score = score

        # If no keywords matched, try to infer from common support patterns
        if best_score <= 0:
            if any(t in text for t in ["how to", "where is", "help with", "question"]):
                best_area = "general"
            else:
                best_area = "technical"

        return ClassificationResult(
            request_type=request_type,
            product_area=best_area,
            justification=f"Classified using keyword heuristics (score={best_score}); request_type={request_type}, product_area={best_area}.",
        )
