"""
Tool / function-calling layer.

Each function here is a "tool" the agent's plan can invoke — the same
concept as LangChain/OpenAI function-calling tools, written explicitly so
the dispatch logic (TOOL_REGISTRY, call_tool) is visible and testable rather
than hidden behind a framework decorator. Every tool makes a real HTTP call
to the mock Order Management API in src/external_service.py (see that file
for why it's a mock rather than a real third-party host in this sandbox).

Each tool returns a uniform ToolResult so the graph's tools_apis node can
handle success and failure the same way regardless of which tool ran —
that uniform error shape is what "error handling and workflow control"
means in practice here, rather than a try/except scattered per call site.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import httpx

EXTERNAL_API_BASE_URL = os.environ.get("EXTERNAL_API_BASE_URL", "http://127.0.0.1:8002")
REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    http_status: Optional[int] = None


def _request(method: str, path: str, **kwargs) -> ToolResult:
    tool_name = kwargs.pop("_tool_name", path)
    url = f"{EXTERNAL_API_BASE_URL}{path}"
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return ToolResult(ok=False, tool=tool_name, error=str(detail), http_status=resp.status_code)
        return ToolResult(ok=True, tool=tool_name, data=resp.json(), http_status=resp.status_code)
    except httpx.ConnectError:
        return ToolResult(
            ok=False,
            tool=tool_name,
            error=(
                "Could not reach the order-management API at "
                f"{EXTERNAL_API_BASE_URL}. Is src/external_service.py running?"
            ),
        )
    except httpx.TimeoutException:
        return ToolResult(ok=False, tool=tool_name, error="Order-management API timed out.")
    except Exception as exc:  # last-resort guard so a bad tool call never crashes the graph
        return ToolResult(ok=False, tool=tool_name, error=f"Unexpected tool error: {exc}")


def check_order_status(order_id: str) -> ToolResult:
    """Look up an order's current status, amount, and shipping address."""
    return _request("GET", f"/orders/{order_id}", _tool_name="check_order_status")


def process_refund(order_id: str, amount: float, reason: str) -> ToolResult:
    """Issue a refund for an order. The mock API itself enforces business
    rules (order must be delivered/shipped, amount can't exceed the order
    total) — the agent's planner also checks eligibility *before* calling
    this, so a rejection here is a defense-in-depth backstop, not the
    primary control."""
    return _request(
        "POST",
        f"/orders/{order_id}/refund",
        json={"amount": amount, "reason": reason},
        _tool_name="process_refund",
    )


def update_shipping_address(order_id: str, new_address: str) -> ToolResult:
    """Update the shipping address on an order that hasn't shipped yet."""
    return _request(
        "POST",
        f"/orders/{order_id}/address",
        json={"new_address": new_address},
        _tool_name="update_shipping_address",
    )


TOOL_REGISTRY: Dict[str, Callable[..., ToolResult]] = {
    "check_order_status": check_order_status,
    "process_refund": process_refund,
    "update_shipping_address": update_shipping_address,
}


def call_tool(tool_name: str, **kwargs) -> ToolResult:
    """Dispatch a named tool call with keyword arguments — the same shape
    the agent's plan (src/planner.py) produces for each step."""
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return ToolResult(ok=False, tool=tool_name, error=f"Unknown tool '{tool_name}'")
    try:
        return tool(**kwargs)
    except TypeError as exc:
        return ToolResult(ok=False, tool=tool_name, error=f"Bad arguments for tool '{tool_name}': {exc}")
