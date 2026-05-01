from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from anthropic import Anthropic
from openai import OpenAI
from pydantic import ValidationError

from .logging_utils import PromptLogger
from .models import ResponsePayload, RetrievedChunk, Ticket


SYSTEM_PROMPT = (
    "You are a support assistant. Use only provided support excerpts. "
    "Do not invent facts. If context is insufficient, say so briefly."
)


@dataclass
class Responder:
    provider: str
    model: str
    logger: PromptLogger

    def _call_llm(self, prompt: str) -> str:
        self.logger.append("PROMPT", prompt)
        provider = self.provider.lower()
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=self.model,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in msg.content if hasattr(block, "text"))
            self.logger.append("RESPONSE", text)
            return text
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            msg = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            text = msg.output_text
            self.logger.append("RESPONSE", text)
            return text
        fallback = json.dumps(
            {
                "response": "I could not generate a model-based response, but based on retrieved docs, please follow your official support workflow.",
                "justification": "LLM key unavailable; fallback grounded reply generated from pipeline safeguards.",
            }
        )
        self.logger.append("RESPONSE", fallback)
        return fallback

    def reply(self, ticket: Ticket, chunks: list[RetrievedChunk], request_type: str, product_area: str) -> ResponsePayload:
        citations = "\n".join(
            f"- [{chunk.chunk_id}] ({chunk.company}) {chunk.source_path}" for chunk in chunks
        )
        context = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks
        )
        prompt = f"""
Ticket:
Subject: {ticket.subject}
Issue: {ticket.issue}
Company: {ticket.company}
Request Type: {request_type}
Product Area: {product_area}

Retrieved Support Excerpts:
{context}

Citations:
{citations}

Return strict JSON with keys:
- response: user-facing support response grounded only in excerpts. Add inline source tags like [chunk_id].
- justification: concise reason referencing the most relevant chunk ids.
"""
        raw = self._call_llm(prompt.strip())
        for _ in range(2):
            try:
                payload = ResponsePayload.model_validate_json(raw)
                return payload
            except ValidationError:
                repair_prompt = (
                    "Reformat the previous answer as strict JSON with keys "
                    "'response' and 'justification' only."
                )
                raw = self._call_llm(repair_prompt)

        return ResponsePayload(
            response="We need to escalate this request to a human agent due to formatting uncertainty.",
            justification="Fallback after JSON validation retry failure.",
        )

    @staticmethod
    def validate_grounding(payload: ResponsePayload, chunks: list[RetrievedChunk]) -> tuple[bool, str]:
        if not chunks:
            return False, "No retrieved support context was available."
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        cited = set(re.findall(r"\[([^\]]+)\]", payload.response))
        valid_cited = cited.intersection(chunk_ids)
        if not valid_cited:
            return False, "Response missing valid chunk citations."

        reference_text = " ".join(chunk.text.lower() for chunk in chunks)
        response_words = {w for w in re.findall(r"[a-zA-Z]{5,}", payload.response.lower())}
        overlap = sum(1 for word in response_words if word in reference_text)
        if response_words and (overlap / max(len(response_words), 1)) < 0.2:
            return False, "Low lexical grounding overlap with retrieved support excerpts."
        return True, "Grounding validation passed."
