from __future__ import annotations

from dataclasses import dataclass

from .classifier import Classifier
from .escalation import EscalationLogic
from .models import ProcessMeta, Ticket, TicketOutput
from .responder import Responder
from .retriever import Retriever


@dataclass
class Pipeline:
    escalation: EscalationLogic
    classifier: Classifier
    retriever: Retriever
    responder: Responder

    def process_ticket(self, ticket: Ticket) -> tuple[TicketOutput, ProcessMeta]:
        classification = self.classifier.classify(ticket)

        direct_risk = self.escalation.evaluate_text_risk(ticket.text())
        if direct_risk.escalated:
            output = TicketOutput(
                status="escalated",
                product_area=classification.product_area,
                response="Your request has been escalated to a specialist support team.",
                justification=f"{direct_risk.reason} {classification.justification}",
                request_type=classification.request_type,
            )
            return output, ProcessMeta(
                company=ticket.company,
                status="escalated",
                confidence=0.0,
                risk_category=direct_risk.category,
                request_type=classification.request_type,
            )

        retrieval = self.retriever.retrieve(ticket)
        conf_risk = self.escalation.evaluate_retrieval_confidence([retrieval.avg_confidence])
        if conf_risk.escalated:
            output = TicketOutput(
                status="escalated",
                product_area=classification.product_area,
                response="We are escalating this request because we could not confidently match support guidance.",
                justification=f"{conf_risk.reason} rewritten_query='{retrieval.rewritten_query}'. {classification.justification}",
                request_type=classification.request_type,
            )
            return output, ProcessMeta(
                company=ticket.company,
                status="escalated",
                confidence=retrieval.avg_confidence,
                risk_category=conf_risk.category,
                request_type=classification.request_type,
            )

        payload = self.responder.reply(
            ticket=ticket,
            chunks=retrieval.chunks,
            request_type=classification.request_type,
            product_area=classification.product_area,
        )
        grounded, grounding_reason = self.responder.validate_grounding(payload, retrieval.chunks)
        if not grounded:
            output = TicketOutput(
                status="escalated",
                product_area=classification.product_area,
                response="We are escalating this request to ensure a fully grounded response from a specialist.",
                justification=f"Escalated: post-generation validation failed ({grounding_reason}). rewritten_query='{retrieval.rewritten_query}'.",
                request_type=classification.request_type,
            )
            return output, ProcessMeta(
                company=ticket.company,
                status="escalated",
                confidence=retrieval.avg_confidence,
                risk_category="post_generation_guardrail",
                request_type=classification.request_type,
            )

        output = TicketOutput(
            status="replied",
            product_area=classification.product_area,
            response=payload.response,
            justification=(
                f"{payload.justification} "
                f"{classification.justification} "
                f"retrieval_confidence={retrieval.avg_confidence:.3f}; rewritten_query='{retrieval.rewritten_query}'."
            ),
            request_type=classification.request_type,
        )
        return output, ProcessMeta(
            company=ticket.company,
            status="replied",
            confidence=retrieval.avg_confidence,
            risk_category="none",
            request_type=classification.request_type,
        )
