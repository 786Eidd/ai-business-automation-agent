"""
Mock third-party Order Management / CRM system.

This is a real, runnable FastAPI service — a stand-in for a system like
Shopify, a CRM, or an internal order-management API. The agent's tool layer
(src/tools.py) talks to it over real HTTP, exactly the way it would talk to
an actual third-party API: same request/response shapes, same error codes,
same statelessness of the client. Swapping this for a real vendor's API
means changing the base URL and auth headers in src/tools.py — nothing about
the agent's reasoning, planning, or graph wiring needs to change.

Why a mock and not a real third-party API: this project was built in a
sandboxed environment with no outbound access to arbitrary third-party
hosts (the same constraint documented in the README for the LLM path, and
in the other two portfolio projects for LLM/model-hub access). Building a
real local HTTP service, rather than a plain in-process function call,
keeps the "API integration" honest and testable: the tool layer genuinely
makes network requests and genuinely has to handle timeouts, 404s, and
validation errors, not just call a Python function directly.

Run standalone:
    uvicorn src.external_service:app --port 8002
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")


def _resolve_data_path() -> str:
    """Allow evaluate.py to point this service at an isolated copy of the
    dataset (env var ORDERS_DATA_PATH), so running the evaluation repeatedly
    doesn't permanently mutate the committed seed data in data/orders.json
    (refunds and address changes are stateful writes)."""
    return os.environ.get("ORDERS_DATA_PATH", _DEFAULT_DATA_PATH)


app = FastAPI(title="Mock Order Management API", version="1.0")

_lock = threading.Lock()


def _load_orders() -> dict:
    with open(_resolve_data_path(), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {o["order_id"]: o for o in payload["orders"]}


def _save_orders(orders_by_id: dict) -> None:
    with open(_resolve_data_path(), "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["orders"] = list(orders_by_id.values())
    with open(_resolve_data_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class RefundRequest(BaseModel):
    amount: float
    reason: str


class AddressUpdateRequest(BaseModel):
    new_address: str


class RefundResponse(BaseModel):
    order_id: str
    refund_id: str
    amount: float
    status: str


class AddressUpdateResponse(BaseModel):
    order_id: str
    new_address: str
    status: str


@app.get("/orders/{order_id}")
def get_order(order_id: str):
    with _lock:
        orders = _load_orders()
    order = orders.get(order_id.upper())
    if order is None:
        raise HTTPException(status_code=404, detail=f"No order found with id '{order_id}'")
    return order


@app.post("/orders/{order_id}/refund", response_model=RefundResponse)
def issue_refund(order_id: str, req: RefundRequest):
    with _lock:
        orders = _load_orders()
        order = orders.get(order_id.upper())
        if order is None:
            raise HTTPException(status_code=404, detail=f"No order found with id '{order_id}'")
        if order["status"] not in ("delivered", "shipped"):
            raise HTTPException(
                status_code=409,
                detail=f"Order '{order_id}' is still '{order['status']}' — nothing to refund yet.",
            )
        if req.amount > order["amount"]:
            raise HTTPException(
                status_code=422,
                detail=f"Refund amount {req.amount} exceeds order total {order['amount']}.",
            )
        order["status"] = "refunded"
        order["refund_amount"] = req.amount
        order["refund_reason"] = req.reason
        orders[order["order_id"]] = order
        _save_orders(orders)
    return RefundResponse(
        order_id=order["order_id"],
        refund_id=f"RFND-{order['order_id'].split('-')[-1]}",
        amount=req.amount,
        status="issued",
    )


@app.post("/orders/{order_id}/address", response_model=AddressUpdateResponse)
def update_address(order_id: str, req: AddressUpdateRequest):
    with _lock:
        orders = _load_orders()
        order = orders.get(order_id.upper())
        if order is None:
            raise HTTPException(status_code=404, detail=f"No order found with id '{order_id}'")
        if order["status"] != "processing":
            raise HTTPException(
                status_code=409,
                detail=f"Order '{order_id}' has already {order['status']} — address can no longer be changed.",
            )
        order["shipping_address"] = req.new_address
        orders[order["order_id"]] = order
        _save_orders(orders)
    return AddressUpdateResponse(order_id=order["order_id"], new_address=req.new_address, status="updated")


@app.get("/health")
def health():
    return {"status": "ok"}
