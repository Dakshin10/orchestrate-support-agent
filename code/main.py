import pandas as pd
import time

from classifier import classify_ticket
from safety import should_escalate
from retriever import Retriever
from responder import generate_response_with_confidence


CONFIDENCE_THRESHOLD = 0.08
RESPONSE_MIN_LENGTH = 20


def process_ticket(ticket_text, retriever):
    start = time.perf_counter()

    classification = classify_ticket(ticket_text)
    product_area = classification["product_area"]
    request_type = classification["request_type"]
    confidence = classification.get("confidence", 0.5)

    if confidence < CONFIDENCE_THRESHOLD:
        return escalate(product_area, request_type,
                        "We could not confidently understand the request. Escalated to support.",
                        confidence, start)

    safety = should_escalate(ticket_text, confidence, product_area)

    if safety["escalate"]:
        return escalate(product_area, request_type,
                        safety["reason"], confidence, start)

    docs = retriever.retrieve(ticket_text, product_area, top_k=3)
    doc_texts = [d[0] for d in docs]

    if not doc_texts:
        return escalate(product_area, request_type,
                        "We could not find sufficient information in our support resources. Escalated to a human agent.",
                        confidence, start)

    response, resp_conf = generate_response_with_confidence(ticket_text, doc_texts)

    if not response or len(response) < RESPONSE_MIN_LENGTH:
        return escalate(product_area, request_type,
                        "We are unable to confidently resolve this issue using available documentation. Escalated to support.",
                        confidence, start)

    return {
        "status": "replied",
        "product_area": product_area,
        "response": response,
        "justification": f"Resolved using {product_area} documentation",
        "request_type": request_type,
        "classification_confidence": round(confidence, 3),
        "response_confidence": round(resp_conf, 3),
        "processing_ms": int((time.perf_counter() - start) * 1000)
    }


def escalate(product_area, request_type, reason, confidence, start):
    return {
        "status": "escalated",
        "product_area": product_area,
        "response": reason,
        "justification": reason,
        "request_type": request_type,
        "classification_confidence": round(confidence, 3),
        "response_confidence": 0.0,
        "processing_ms": int((time.perf_counter() - start) * 1000)
    }


def main():
    df = pd.read_csv("support_tickets/support_tickets.csv")
    retriever = Retriever(data_path="data")

    results = []

    for i, row in df.iterrows():
        ticket = str(row.iloc[0])
        print(f"[{i+1}] Processing...")

        result = process_ticket(ticket, retriever)
        results.append(result)

    output = pd.DataFrame(results)

    output.to_csv("support_tickets/output.csv", index=False)

    print("\n✅ output.csv updated successfully!")


if __name__ == "__main__":
    main()