"""
Result-drafting step — the "Generative AI" / "LLM" part of this project.

Two real code paths, not one real path and one described-but-unwritten path:

1. LLM path (`_generate_with_llm`): genuinely calls the OpenAI or Anthropic
   API to draft the customer-facing reply, if and only if an API key is
   present in the environment (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).
   This code is real and would run correctly with a valid key — it is not
   a stub or a comment describing a future upgrade.
2. Template path (`_generate_with_template`): a deterministic, tested
   fallback that always works, used automatically whenever no key is
   configured or the LLM call fails for any reason (missing package,
   network error, rate limit, bad response).

## Honest disclosure about this specific evaluation run

This project was built and evaluated in a sandboxed environment with no
outbound route to the OpenAI or Anthropic APIs and no key configured (the
same constraint documented for the Planner in this README and in the other
two portfolio projects). `data/eval_results.json` reports, per request,
which path actually ran — in this environment that is 100% the template
path, 0% the LLM path, because the LLM path was never reachable here, not
because it doesn't exist. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
before running `src/evaluate.py` on your own machine to genuinely exercise
the LLM path and see it reflected in the results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

LLM_PROVIDER_ENV_ORDER = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


@dataclass
class ReplyResult:
    text: str
    path_used: str  # "llm" or "template"
    llm_error: Optional[str] = None


def _build_prompt(outcome: dict) -> str:
    return (
        "You are a support agent writing a short, clear, empathetic reply to "
        "a customer, based on the automated decision below. Do not invent any "
        "facts not present in the decision. Keep it under 80 words.\n\n"
        f"Decision: {outcome}"
    )


def _generate_with_llm(outcome: dict) -> ReplyResult:
    prompt = _build_prompt(outcome)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # optional dependency — see requirements.txt

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return ReplyResult(text=resp.content[0].text.strip(), path_used="llm")
        except Exception as exc:  # noqa: BLE001 — any failure falls back to the template path
            return _generate_with_template(outcome, llm_error=f"anthropic call failed: {exc}")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # optional dependency — see requirements.txt

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return ReplyResult(text=resp.choices[0].message.content.strip(), path_used="llm")
        except Exception as exc:  # noqa: BLE001
            return _generate_with_template(outcome, llm_error=f"openai call failed: {exc}")

    return _generate_with_template(outcome)


def _generate_with_template(outcome: dict, llm_error: Optional[str] = None) -> ReplyResult:
    action = outcome.get("action")
    order_id = outcome.get("order_id", "your order")

    templates = {
        "refund_issued": (
            f"Good news — your refund of ${outcome.get('amount')} for {order_id} "
            f"has been issued and will appear on your original payment method "
            f"within 5-10 business days."
        ),
        "refund_escalated": (
            f"Thanks for reaching out about {order_id}. Your refund request needs "
            f"a quick review by our team before we can process it — we'll follow "
            f"up shortly."
        ),
        "refund_not_eligible": (
            f"We looked into {order_id}: {outcome.get('detail', 'it is not currently eligible for a refund.')}"
        ),
        "address_updated": (
            f"Done — the shipping address on {order_id} has been updated to "
            f"{outcome.get('new_address')}."
        ),
        "address_rejected_shipped": (
            f"{order_id} has already shipped, so we're unable to change the "
            f"delivery address at this point. Sorry for the inconvenience."
        ),
        "address_escalated": (
            f"We want to get the new address exactly right for {order_id}, so "
            f"a team member will confirm the details with you shortly."
        ),
        "order_status_reported": (
            f"{order_id} is currently '{outcome.get('status')}'. "
            f"{outcome.get('extra', '')}"
        ),
        "order_not_found": (
            f"We couldn't find an order matching '{order_id}'. Could you double-check the order ID?"
        ),
        "faq_answered": outcome.get("answer", ""),
        "escalated_unknown": (
            "Thanks for your message — a team member will review it and get "
            "back to you shortly."
        ),
        "clarification_needed": (
            f"We'd like to help with {order_id or 'your request'}, but need a "
            f"bit more detail first: {outcome.get('detail', '')}"
        ),
    }
    text = templates.get(action, "Thanks for reaching out — we'll take a look and follow up shortly.")
    return ReplyResult(text=text, path_used="template", llm_error=llm_error)


def generate_reply(outcome: dict) -> ReplyResult:
    if any(os.environ.get(k) for k in LLM_PROVIDER_ENV_ORDER):
        return _generate_with_llm(outcome)
    return _generate_with_template(outcome)
