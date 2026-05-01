from __future__ import annotations

import os
import re
from dataclasses import dataclass

from anthropic import Anthropic
from openai import OpenAI

from .logging_utils import PromptLogger
from .models import Ticket


def _deterministic_rewrite(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.replace("acnt", "account").replace("pwd", "password").replace("txn", "transaction")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class QueryRewriter:
    provider: str
    model: str
    logger: PromptLogger
    enabled: bool = True

    def rewrite(self, ticket: Ticket) -> str:
        raw_text = ticket.text()
        fallback = _deterministic_rewrite(raw_text)
        if not self.enabled:
            return fallback

        prompt = (
            "Rewrite the support ticket into a concise retrieval query. "
            "Keep product names, error codes, and entities. Output plain text only.\n\n"
            f"Subject: {ticket.subject}\nIssue: {ticket.issue}\nCompany: {ticket.company}"
        )
        self.logger.append("QUERY_REWRITE_PROMPT", prompt)
        provider = self.provider.lower()
        try:
            if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
                client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=120,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                rewritten = "".join(block.text for block in msg.content if hasattr(block, "text")).strip()
                self.logger.append("QUERY_REWRITE_RESPONSE", rewritten)
                return rewritten or fallback
            if os.getenv("OPENAI_API_KEY"):
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                msg = client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                rewritten = (msg.output_text or "").strip()
                self.logger.append("QUERY_REWRITE_RESPONSE", rewritten)
                return rewritten or fallback
        except Exception as exc:
            self.logger.append("QUERY_REWRITE_ERROR", str(exc))
        return fallback
