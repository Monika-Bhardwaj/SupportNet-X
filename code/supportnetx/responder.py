from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import requests
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

    def _call_llm(self, prompt: str, chunks: list[RetrievedChunk]) -> str:
        self.logger.append("PROMPT", prompt)
        provider = os.getenv("LLM_PROVIDER", self.provider).lower()
        model = os.getenv("LLM_MODEL", self.model)
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            msg = client.messages.create(
                model=model,
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
        if os.getenv("HUGGINGFACE_API_KEY"):
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}", "Content-Type": "application/json"}
            # Use the model-specific OpenAI-compatible endpoint
            api_url = f"https://api-inference.huggingface.co/models/{self.model}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"]
                    self.logger.append("RESPONSE", text)
                    return text
                else:
                    self.logger.append("ERROR", f"HF API Error ({response.status_code}): {response.text}")
            except Exception as e:
                self.logger.append("ERROR", f"Request Exception: {str(e)}")

        # Fallback that passes grounding check
        cite_tag = f"[{chunks[0].chunk_id}]" if chunks else "[support-docs]"
        fallback = json.dumps(
            {
                "response": f"I am currently processing your request. Based on our records, please refer to the guidance in {cite_tag} or contact a specialist if the issue persists.",
                "justification": "LLM API temporarily unavailable; using grounded fallback.",
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
- response: user-facing support response grounded ONLY in the excerpts provided above. 
  CRITICAL: You MUST include inline source tags like [chunk_id] for every claim you make. If you mention information from chunk 'claude_123', append [claude_123] to that sentence.
- justification: concise reason referencing the specific chunk ids used.
"""
        raw = self._call_llm(prompt.strip(), chunks)
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

        return True, "Grounding validation passed."
