"""
Wires the full workflow as an actual LangGraph StateGraph, in exactly the
order described in this project's portfolio listing:

    User Request -> AI Agent -> Reasoning & Planning -> Knowledge/Data
    Retrieval -> Tools & APIs -> Business Action -> Result / Human Review

Each stage below is one graph node operating on a shared typed state. The
business rules that decide auto-resolve vs. escalate-to-a-human live in
`tools_apis_node`, since that's the point where the agent has both the plan
(what the customer wants) and the retrieved context (what's actually true
about the order) needed to make that call.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import planner, reply_generator, retriever, tools

# Refunds at or below this amount, for an eligible order, are auto-approved.
# Above it, a human reviews before anything is refunded — a deliberately
# simple, explicit policy rather than a hidden heuristic.
REFUND_AUTO_APPROVE_THRESHOLD = 100.0

_DAMAGE_KEYWORDS = ("damaged", "defective", "broken", "faulty")


class GraphState(TypedDict, total=False):
    request_text: str
    intent: str
    confidence: float
    rule_matched: str
    order_id: Optional[str]
    amount: Optional[float]
    new_address: Optional[str]
    needs_clarification: bool
    clarification_reason: Optional[str]
    order_context: Optional[Dict[str, Any]]
    order_lookup_error: Optional[str]
    faq_answer: Optional[str]
    faq_score: Optional[float]
    action: str
    human_review: bool
    detail: Optional[str]
    reply: str
    reply_path: str


def _within_return_window(order: Dict[str, Any]) -> bool:
    if not order.get("delivered_date"):
        return True  # not yet delivered — window hasn't started counting down against the customer
    delivered = _dt.date.fromisoformat(order["delivered_date"])
    window_end = delivered + _dt.timedelta(days=order.get("return_window_days", 30))
    return _dt.date.today() <= window_end


def reasoning_planning_node(state: GraphState) -> GraphState:
    result = planner.plan(state["request_text"])
    return {
        **state,
        "intent": result.intent,
        "confidence": result.confidence,
        "rule_matched": result.rule_matched,
        "order_id": result.order_id,
        "amount": result.amount,
        "new_address": result.new_address,
        "needs_clarification": result.needs_clarification,
        "clarification_reason": result.clarification_reason,
    }


def knowledge_retrieval_node(state: GraphState) -> GraphState:
    updates: GraphState = {}
    if state["intent"] in (planner.ORDER_STATUS, planner.REFUND_REQUEST, planner.ADDRESS_UPDATE) and state.get("order_id"):
        res = retriever.retrieve_order_context(state["order_id"])
        if res.ok:
            updates["order_context"] = res.data
            updates["order_lookup_error"] = None
        else:
            updates["order_context"] = None
            updates["order_lookup_error"] = res.error
    elif state["intent"] == planner.FAQ_QUESTION:
        hit = retriever.retrieve_faq_answer(state["request_text"])
        updates["faq_answer"] = hit.answer if hit else None
        updates["faq_score"] = hit.score if hit else None
    return {**state, **updates}


def tools_apis_node(state: GraphState) -> GraphState:
    intent = state["intent"]
    order_id = state.get("order_id")

    if state.get("needs_clarification") and intent != planner.ADDRESS_UPDATE:
        # ADDRESS_UPDATE with a missing address is handled below (order lookup
        # already happened); every other "needs clarification" case (missing
        # order id, or an unrecognized intent) can't proceed to a tool call.
        action = "clarification_needed" if intent != planner.UNKNOWN else "escalated_unknown"
        return {**state, "action": action, "human_review": True, "detail": state.get("clarification_reason")}

    if intent == planner.ORDER_STATUS:
        if not state.get("order_context"):
            return {**state, "action": "order_not_found", "human_review": False, "detail": state.get("order_lookup_error")}
        order = state["order_context"]
        return {
            **state,
            "action": "order_status_reported",
            "human_review": False,
            "detail": f"Order date {order['order_date']}.",
        }

    if intent == planner.REFUND_REQUEST:
        if not state.get("order_context"):
            return {**state, "action": "order_not_found", "human_review": False, "detail": state.get("order_lookup_error")}
        order = state["order_context"]
        if order["status"] not in ("delivered", "shipped"):
            return {
                **state, "action": "refund_not_eligible", "human_review": False,
                "detail": f"Order is still '{order['status']}' — nothing to refund yet.",
            }
        amount = state.get("amount") or order["amount"]
        is_damage_claim = any(k in state["request_text"].lower() for k in _DAMAGE_KEYWORDS)
        eligible_window = is_damage_claim or _within_return_window(order)

        if amount > REFUND_AUTO_APPROVE_THRESHOLD:
            return {
                **state, "action": "refund_escalated", "human_review": True, "amount": amount,
                "detail": f"Refund amount ${amount} exceeds the ${REFUND_AUTO_APPROVE_THRESHOLD} auto-approval threshold.",
            }
        if not eligible_window:
            return {
                **state, "action": "refund_escalated", "human_review": True, "amount": amount,
                "detail": "Return window has expired and no damage/defect was reported — needs manual review.",
            }

        result = tools.process_refund(order_id, amount, reason=state["request_text"])
        if result.ok:
            return {**state, "action": "refund_issued", "human_review": False, "amount": amount}
        return {
            **state, "action": "refund_escalated", "human_review": True, "amount": amount,
            "detail": f"Refund API call failed, routed to a human: {result.error}",
        }

    if intent == planner.ADDRESS_UPDATE:
        if not order_id:
            return {
                **state, "action": "clarification_needed", "human_review": True,
                "detail": state.get("clarification_reason") or "No order ID found in the request.",
            }
        if not state.get("new_address"):
            return {
                **state, "action": "address_escalated", "human_review": True,
                "detail": state.get("clarification_reason") or "Could not confidently parse the new address.",
            }
        if not state.get("order_context"):
            return {**state, "action": "order_not_found", "human_review": False, "detail": state.get("order_lookup_error")}
        order = state["order_context"]
        if order["status"] != "processing":
            return {
                **state, "action": "address_rejected_shipped", "human_review": False,
                "detail": f"Order has already {order['status']}.",
            }
        result = tools.update_shipping_address(order_id, state["new_address"])
        if result.ok:
            return {**state, "action": "address_updated", "human_review": False}
        return {
            **state, "action": "address_escalated", "human_review": True,
            "detail": f"Address-update API call failed, routed to a human: {result.error}",
        }

    if intent == planner.FAQ_QUESTION:
        if not state.get("faq_answer"):
            return {
                **state, "action": "escalated_unknown", "human_review": True,
                "detail": "No confident FAQ match found for this question.",
            }
        return {**state, "action": "faq_answered", "human_review": False}

    return {**state, "action": "escalated_unknown", "human_review": True, "detail": "Unrecognized request."}


def business_action_node(state: GraphState) -> GraphState:
    """Placeholder for durable side effects beyond the tool call itself
    (e.g. writing an audit-log entry, notifying a Slack channel). The tool
    call that actually changes state (refund/address update) already
    happened in tools_apis_node against the mock Order Management API —
    this node is where that gets recorded for the business, not where the
    action is decided. Kept as an explicit graph node (rather than folded
    into tools_apis_node) because in a real deployment this is genuinely a
    separate concern with its own failure modes (e.g. the audit log write
    failing shouldn't undo a refund that already succeeded)."""
    log_entry = {
        "order_id": state.get("order_id"),
        "intent": state["intent"],
        "action": state["action"],
        "human_review": state["human_review"],
    }
    return {**state, "_audit_log_entry": log_entry}  # type: ignore[typeddict-item]


def result_human_review_node(state: GraphState) -> GraphState:
    outcome = {
        "action": state["action"],
        "order_id": state.get("order_id"),
        "amount": state.get("amount"),
        "new_address": state.get("new_address"),
        "status": state.get("order_context", {}).get("status") if state.get("order_context") else None,
        "answer": state.get("faq_answer"),
        "detail": state.get("detail"),
        "extra": state.get("detail") or "",
    }
    reply = reply_generator.generate_reply(outcome)
    return {**state, "reply": reply.text, "reply_path": reply.path_used}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("reasoning_planning", reasoning_planning_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("tools_apis", tools_apis_node)
    graph.add_node("business_action", business_action_node)
    graph.add_node("result_human_review", result_human_review_node)

    graph.set_entry_point("reasoning_planning")
    graph.add_edge("reasoning_planning", "knowledge_retrieval")
    graph.add_edge("knowledge_retrieval", "tools_apis")
    graph.add_edge("tools_apis", "business_action")
    graph.add_edge("business_action", "result_human_review")
    graph.add_edge("result_human_review", END)

    return graph.compile()


def run(request_text: str) -> GraphState:
    app_graph = build_graph()
    return app_graph.invoke({"request_text": request_text})
