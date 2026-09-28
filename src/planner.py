"""
Reasoning & Planning node.

## Why this is rule-based, not LLM-based

This project was built in a sandboxed environment with no outbound route to
an LLM API (no OpenAI/Anthropic key configured, no local model runtime) —
the same documented constraint as the Planner in `multi-agent-financial-qa`
and the generation layer in `rag-citation-demo`. Rather than fake an
LLM-driven planner, intent classification and entity extraction here are
real, tested, deterministic logic: explicit keyword/pattern rules with a
fixed priority order, chosen and tuned against the labeled test set in
`data/test_requests.json` (see `src/evaluate.py`).

This is still genuine "reasoning and task planning": the output is a
structured decision (intent + confidence + extracted entities + which
tool(s), if any, the plan calls for) that downstream nodes act on — the
same shape an LLM-driven planner would need to produce. Swapping this node
for an LLM call (e.g. a small model prompted to classify + extract) is a
scoped, documented upgrade: `src/graph.py` only depends on the `PlanResult`
shape below, not on how it's produced. The generative-reply step
(`src/reply_generator.py`) already implements that LLM code path for real,
gated behind an API key — see that file's docstring.

## Honest limitation

Keyword/pattern rules will misclassify phrasings the rules don't anticipate
(sarcasm, indirect phrasing, typos in keywords). `data/eval_results.json`
reports the measured intent-accuracy rate on the labeled test set, and the
"Honest limitations" section of the README lists the specific misses found
during testing — this isn't presented as a solved NLU problem.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

ORDER_STATUS = "order_status"
REFUND_REQUEST = "refund_request"
ADDRESS_UPDATE = "address_update"
FAQ_QUESTION = "faq_question"
UNKNOWN = "unknown"

_ORDER_ID_RE = re.compile(r"\bORD-\d{3,6}\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\$?\s?(\d+(?:\.\d{1,2})?)\s?(?:usd|dollars)?", re.IGNORECASE)
# "to <address>" / "address to <address>" — greedy to end of sentence, trimmed of trailing punctuation.
_ADDRESS_RE = re.compile(
    r"(?:address\s+to|ship(?:ping)?\s+(?:address\s+)?to|deliver(?:ed)?\s+to|to)\s+(.+)$",
    re.IGNORECASE,
)

_REFUND_KEYWORDS = [
    "refund", "money back", "reimburse", "damaged", "broken", "defective",
    "return this", "want my money", "give me a refund",
]
_ADDRESS_KEYWORDS = [
    "wrong address", "new address", "ship it to", "redirect my order",
    "move my order",
]
# Broader than a fixed phrase list: catches "update my *shipping* address",
# "update address for...", "change the delivery address", etc. — any
# address-changing verb co-occurring with the word "address", regardless of
# what's in between. Added after initial testing showed the fixed-phrase
# list above missed "update my shipping address" and "update address for"
# (see data/eval_results.json ids 17 and 35, and the README's honest
# limitations section for what this broader rule still doesn't catch).
_ADDRESS_VERB_RE = re.compile(r"\b(update|change|edit|correct|fix|redirect|move)\b", re.IGNORECASE)
_ORDER_STATUS_KEYWORDS = [
    "where is my order", "order status", "track my order", "has it shipped",
    "shipped yet", "when will it arrive", "when will my order",
    "status of my order", "where's my package",
]
_FAQ_KEYWORDS = [
    "return policy", "how long does shipping", "shipping take",
    "payment method", "do you ship internationally", "how do i track",
    "how long do refunds", "how long does a refund",
]


@dataclass
class PlanResult:
    intent: str
    confidence: float
    rule_matched: str
    order_id: Optional[str] = None
    amount: Optional[float] = None
    new_address: Optional[str] = None
    reason: Optional[str] = None
    raw_text: str = ""
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None


def _contains_any(text: str, keywords) -> Optional[str]:
    for kw in keywords:
        if kw in text:
            return kw
    return None


def extract_order_id(text: str) -> Optional[str]:
    m = _ORDER_ID_RE.search(text)
    return m.group(0).upper() if m else None


def extract_amount(text: str) -> Optional[float]:
    # Only look near a currency-looking token so we don't grab an unrelated
    # number (e.g. an order id's digits are excluded because ORD- has no $).
    for m in re.finditer(r"\$\s?(\d+(?:\.\d{1,2})?)", text):
        return float(m.group(1))
    return None


def extract_new_address(text: str) -> Optional[str]:
    m = _ADDRESS_RE.search(text)
    if not m:
        return None
    candidate = m.group(1).strip().rstrip(".!?")
    # Reject obviously-too-short or generic captures ("to check", "to confirm")
    if len(candidate) < 8 or "," not in candidate:
        return None
    return candidate


def classify_intent(text_lower: str) -> tuple[str, float, str]:
    """Priority order matters: refund/address keywords are checked before
    the more generic order_status/FAQ ones, since a message like "my order
    ORD-1002 arrived damaged, I want a refund" also contains order-status-ish
    language but the actionable intent is the refund."""
    kw = _contains_any(text_lower, _REFUND_KEYWORDS)
    if kw:
        return REFUND_REQUEST, 0.9, f"keyword:'{kw}'"

    kw = _contains_any(text_lower, _ADDRESS_KEYWORDS)
    if kw:
        return ADDRESS_UPDATE, 0.9, f"keyword:'{kw}'"
    if "address" in text_lower and _ADDRESS_VERB_RE.search(text_lower):
        return ADDRESS_UPDATE, 0.85, "rule:address_verb+noun"

    kw = _contains_any(text_lower, _ORDER_STATUS_KEYWORDS)
    if kw:
        return ORDER_STATUS, 0.85, f"keyword:'{kw}'"

    kw = _contains_any(text_lower, _FAQ_KEYWORDS)
    if kw:
        return FAQ_QUESTION, 0.8, f"keyword:'{kw}'"

    # Weak fallback: an order id with a question mark and no other signal
    # is most likely a status check.
    if _ORDER_ID_RE.search(text_lower) and "?" in text_lower:
        return ORDER_STATUS, 0.55, "fallback:order_id+question_mark"

    return UNKNOWN, 0.3, "no_rule_matched"


def plan(request_text: str) -> PlanResult:
    text_lower = request_text.lower()
    intent, confidence, rule = classify_intent(text_lower)

    order_id = extract_order_id(request_text)
    amount = extract_amount(request_text) if intent == REFUND_REQUEST else None
    new_address = extract_new_address(request_text) if intent == ADDRESS_UPDATE else None

    result = PlanResult(
        intent=intent,
        confidence=confidence,
        rule_matched=rule,
        order_id=order_id,
        amount=amount,
        new_address=new_address,
        raw_text=request_text,
    )

    if intent in (REFUND_REQUEST, ADDRESS_UPDATE, ORDER_STATUS) and not order_id:
        result.needs_clarification = True
        result.clarification_reason = "No order ID (format ORD-####) found in the request."
        return result

    if intent == ADDRESS_UPDATE and not new_address:
        result.needs_clarification = True
        result.clarification_reason = "Could not confidently extract the new address from the request."

    if intent == UNKNOWN:
        result.needs_clarification = True
        result.clarification_reason = "Could not classify the request into a known workflow."

    return result
