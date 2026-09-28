"""
Knowledge / Data Retrieval node.

Two kinds of context this agent needs before it can act, both real and
locally computed (no network dependency, no fabricated "vector search"
theater over data that was never actually indexed):

1. Order context — a read-only lookup against the same mock Order
   Management API the Tools & APIs step writes to (src/external_service.py),
   reused here via src/tools.check_order_status. Read vs. write is the
   real distinction between this node and the Tools & APIs node: retrieval
   only ever reads.
2. FAQ knowledge — BM25 (Okapi) lexical search over data/faq_knowledge.json,
   the same ranking algorithm used in the multi-agent-financial-qa and
   rag-citation-demo retrievers, for a consistent, comparable approach
   across the portfolio. A confidence threshold decides whether the best
   match is good enough to answer from directly, or whether the request
   should be handed to a human instead of guessing.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from rank_bm25 import BM25Okapi

from . import tools

FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faq_knowledge.json")

# Below this BM25 score, treat the FAQ match as "not confident enough" and
# escalate rather than answer — the same refuse-rather-than-guess discipline
# as the rag-citation-demo project, applied here to a much smaller corpus.
FAQ_CONFIDENCE_THRESHOLD = 0.5


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class FaqHit:
    faq_id: str
    question: str
    answer: str
    score: float


class FaqIndex:
    def __init__(self):
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.faqs = payload["faqs"]
        corpus = [_tokenize(f"{f['question']} {f['answer']}") for f in self.faqs]
        self._bm25 = BM25Okapi(corpus)

    def query(self, question: str, top_k: int = 1) -> List[FaqHit]:
        scores = self._bm25.get_scores(_tokenize(question))
        ranked = sorted(range(len(self.faqs)), key=lambda i: -scores[i])[:top_k]
        return [
            FaqHit(
                faq_id=self.faqs[i]["id"],
                question=self.faqs[i]["question"],
                answer=self.faqs[i]["answer"],
                score=float(scores[i]),
            )
            for i in ranked
        ]


_faq_index: Optional[FaqIndex] = None


def get_faq_index() -> FaqIndex:
    global _faq_index
    if _faq_index is None:
        _faq_index = FaqIndex()
    return _faq_index


def retrieve_faq_answer(question: str) -> Optional[FaqHit]:
    hit = get_faq_index().query(question, top_k=1)[0]
    if hit.score < FAQ_CONFIDENCE_THRESHOLD:
        return None
    return hit


def retrieve_order_context(order_id: str) -> tools.ToolResult:
    """Read-only order lookup, reusing the same tool the Tools & APIs node
    calls for consistency, but conceptually this is context-gathering, not
    a business action — nothing here mutates state."""
    return tools.check_order_status(order_id)
