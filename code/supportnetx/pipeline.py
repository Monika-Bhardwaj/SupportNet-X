from __future__ import annotations

from dataclasses import dataclass

from .classifier import Classifier
from .escalation import EscalationLogic
from .models import Ticket, TicketOutput
from .responder import Responder
from .retriever import Retriever


@dataclass
class Pipeline:
    escalation: EscalationLogic
    classifier: Classifier
    retriever: Retriever
    responder: Responder

    def process_ticket(self, ticket: Ticket) -> TicketOutput:
        classification = self.classifier.classify(ticket)

        direct_risk = self.escalation.evaluate_text_risk(ticket.text())
        if direct_risk.escalated:
            return TicketOutput(
                status="escalated",
                product_area=classification.product_area,
                response="Your request has been escalated to a specialist support team.",
                justification=f"{direct_risk.reason} {classification.justification}",
                request_type=classification.request_type,
            )

        chunks = self.retriever.retrieve(ticket)
        confidence = self.retriever.confidence(chunks)
        conf_risk = self.escalation.evaluate_retrieval_confidence([confidence])
        if conf_risk.escalated:
            return TicketOutput(
                status="escalated",
                product_area=classification.product_area,
                response="We are escalating this request because we could not confidently match support guidance.",
                justification=f"{conf_risk.reason} {classification.justification}",
                request_type=classification.request_type,
            )

        payload = self.responder.reply(
            ticket=ticket,
            chunks=chunks,
            request_type=classification.request_type,
            product_area=classification.product_area,
        )
        return TicketOutput(
            status="replied",
            product_area=classification.product_area,
            response=payload.response,
            justification=f"{payload.justification} {classification.justification}",
            request_type=classification.request_type,
        )
