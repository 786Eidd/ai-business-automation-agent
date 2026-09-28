"""
Runs every request in data/test_requests.json through the real graph and
writes data/eval_results.json from the actual output — every number in
this project's README traces back to a run of this script.

Two things are set up before anything else is imported, because
src/tools.py and src/external_service.py read their configuration from
environment variables at import time:

1. ORDERS_DATA_PATH points the mock Order Management API at a throwaway
   copy of data/orders.json, so repeated evaluation runs don't permanently
   mutate the committed seed data (refunds and address changes are
   stateful writes against that file).
2. EXTERNAL_API_BASE_URL points the agent's tools at a test instance of
   that API, started in a background thread by this script on a port that
   won't collide with the ports used for local manual runs (8001/8002 —
   see README "Running it").

Run:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_PORT = 8099

_tmp_dir = tempfile.mkdtemp(prefix="ai_biz_agent_eval_")
_tmp_orders_path = str(Path(_tmp_dir) / "orders.json")
shutil.copyfile(DATA_DIR / "orders.json", _tmp_orders_path)

os.environ["ORDERS_DATA_PATH"] = _tmp_orders_path
os.environ["EXTERNAL_API_BASE_URL"] = f"http://127.0.0.1:{EVAL_PORT}"

import uvicorn  # noqa: E402

from . import external_service, graph  # noqa: E402 — must import after env vars are set


def _start_external_service() -> uvicorn.Server:
    config = uvicorn.Config(external_service.app, host="127.0.0.1", port=EVAL_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{EVAL_PORT}/health", timeout=0.5) as resp:
                if resp.status == 200:
                    return server
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Mock external service did not become healthy in time.")


def run_evaluation() -> dict:
    server = _start_external_service()
    try:
        with open(DATA_DIR / "test_requests.json", "r", encoding="utf-8") as f:
            test_cases = json.load(f)["requests"]

        compiled_graph = graph.build_graph()

        records = []
        latencies_ms = []
        errors = 0
        intent_correct = 0
        escalate_correct = 0
        llm_path_count = 0

        for case in test_cases:
            start = time.perf_counter()
            error_text = None
            try:
                result = compiled_graph.invoke({"request_text": case["request_text"]})
            except Exception as exc:  # the thing "error handling" claims are actually tested against
                errors += 1
                error_text = f"{type(exc).__name__}: {exc}"
                result = {
                    "intent": "ERROR", "action": "ERROR", "human_review": True,
                    "reply": "", "reply_path": "error", "order_id": None, "confidence": 0.0,
                }
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

            intent_ok = result["intent"] == case["expected_intent"]
            escalate_ok = result["human_review"] == case["expected_should_escalate"]
            intent_correct += int(intent_ok)
            escalate_correct += int(escalate_ok)
            if result.get("reply_path") == "llm":
                llm_path_count += 1

            records.append({
                "id": case["id"],
                "request_text": case["request_text"],
                "expected_intent": case["expected_intent"],
                "actual_intent": result["intent"],
                "intent_correct": intent_ok,
                "expected_should_escalate": case["expected_should_escalate"],
                "actual_should_escalate": result["human_review"],
                "escalate_correct": escalate_ok,
                "action": result["action"],
                "reply": result["reply"],
                "reply_path": result["reply_path"],
                "latency_ms": round(elapsed_ms, 3),
                "note": case.get("note"),
                "error": error_text,
            })

        n = len(test_cases)
        latencies_sorted = sorted(latencies_ms)
        p95_index = max(0, int(round(0.95 * (n - 1))))

        summary = {
            "total_requests": n,
            "intent_accuracy": round(intent_correct / n, 4),
            "intent_correct": intent_correct,
            "escalation_decision_accuracy": round(escalate_correct / n, 4),
            "escalation_correct": escalate_correct,
            "error_count": errors,
            "error_rate": round(errors / n, 4),
            "llm_reply_path_used_count": llm_path_count,
            "llm_reply_path_used_pct": round(llm_path_count / n, 4),
            "latency_ms_avg": round(statistics.mean(latencies_ms), 3),
            "latency_ms_p95": round(latencies_sorted[p95_index], 3),
            "by_intent": {},
        }

        for intent in sorted({r["expected_intent"] for r in records}):
            subset = [r for r in records if r["expected_intent"] == intent]
            summary["by_intent"][intent] = {
                "count": len(subset),
                "intent_accuracy": round(sum(r["intent_correct"] for r in subset) / len(subset), 4),
                "escalation_decision_accuracy": round(sum(r["escalate_correct"] for r in subset) / len(subset), 4),
            }

        output = {"summary": summary, "records": records}
        with open(DATA_DIR / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(json.dumps(summary, indent=2))
        return output
    finally:
        server.should_exit = True
        shutil.rmtree(_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_evaluation()
