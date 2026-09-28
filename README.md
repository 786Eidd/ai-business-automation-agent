# AI Business Automation Agent | Intelligent Workflow & Task Automation

**Author:** Eid Muhammad — AI Engineer

A working AI agent that turns a repetitive support/operations workflow into
an intelligent, action-oriented process — not a chatbot that only answers
questions. Given a free-text request, the agent reasons about what's being
asked, retrieves the relevant knowledge/order context, calls the right
tool or API, takes the business action, and either resolves it directly or
hands it to a human when the decision needs one — orchestrated as an
actual [LangGraph](https://github.com/langchain-ai/langgraph) state graph,
the same Planner/Retriever/Executor-family pattern behind my
[`multi-agent-financial-qa`](../multi-agent-financial-qa) project.

The concrete workflow this build automates end-to-end: **customer support
request triage** — order status checks, refund requests, shipping-address
changes, and FAQ questions — against a real order-management API this
project also implements (as a mock third-party system; see below).

Every number in this README came from running `python -m src.evaluate` —
see [`data/eval_results.json`](data/eval_results.json) for the raw,
per-request output.

## How the workflow works

```
User Request
   |
   v
AI Agent
   |
   v
Reasoning & Planning     (src/planner.py)
   |
   v
Knowledge / Data Retrieval   (src/retriever.py)
   |
   v
Tools & APIs              (src/tools.py -> src/external_service.py)
   |
   v
Business Action           (src/graph.py: business_action_node)
   |
   v
Result / Human Review     (src/reply_generator.py + escalation queue)
```

Each stage is a real node in a `langgraph.graph.StateGraph` (`src/graph.py`)
— a typed state object is passed through and updated at every step, the
same orchestration model used for LLM-driven LangGraph agents.

## Key capabilities, and where each one actually lives in the code

- **AI agent reasoning and task planning** — `src/planner.py`: classifies
  the request into an intent (order status / refund / address change / FAQ
  / unrecognized), extracts entities (order ID, amount, new address), and
  flags when it can't proceed without clarification.
- **Tool and function calling** — `src/tools.py`: a real `TOOL_REGISTRY`
  dispatch table (`check_order_status`, `process_refund`,
  `update_shipping_address`), each returning a uniform `ToolResult` so
  success/failure is handled the same way regardless of which tool ran.
- **API integrations** — the tools above make genuine HTTP calls (via
  `httpx`) to `src/external_service.py`, a real, separately-runnable
  FastAPI service standing in for a third-party order-management/CRM
  system. See "Why a mock API" below for why it's not a live third-party
  host in this build.
- **Business workflow automation** — `src/graph.py`'s `tools_apis_node`
  encodes the actual business rules (refund auto-approval threshold,
  return-window and damaged-item logic, shipped-order address-lock) that
  decide what happens next.
- **Context and knowledge retrieval** — `src/retriever.py`: read-only order
  lookups plus a BM25 FAQ search over `data/faq_knowledge.json`, the same
  ranking algorithm used in `rag-citation-demo` and
  `multi-agent-financial-qa`.
- **Multi-step task execution** — a single request flows through five
  distinct graph nodes before a result is returned; refund/address requests
  additionally trigger a real API call as part of that flow.
- **Human-in-the-loop handling where required** — high-value refunds
  (over $100), refunds outside the return window without a damage claim,
  unparseable address changes, low-confidence FAQ matches, and
  unclassifiable requests are all routed to `status: "pending_human_review"`
  instead of being auto-resolved.
- **Error handling and workflow control** — invalid order IDs, unreachable
  API calls, and bad tool arguments all return a structured result instead
  of crashing; `src/evaluate.py` runs 35 requests, including several with
  invalid order IDs, and measures **0 unhandled errors**.
- **Integration with external systems and services** — the tools/API layer
  is architected so swapping `src/external_service.py`'s base URL for a
  real CRM's endpoint requires no change to the planner, retriever, or
  graph.

## Why the Planner is rule-based, and how the "LLM" / "Generative AI" piece is real anyway

This project was built in a sandboxed environment with no outbound route to
an LLM API (no OpenAI/Anthropic key, no local model runtime) — the same
constraint documented in `multi-agent-financial-qa`. So `src/planner.py`'s
intent classification is real, tested, deterministic rule logic, not an
LLM call. That's genuine reasoning and planning either way: the output is
a structured decision downstream nodes act on, and swapping this node for
an LLM-driven one later doesn't require changing anything else in the graph.

What *is* real LLM-integration code, not just a described upgrade path, is
`src/reply_generator.py`: it genuinely calls the OpenAI or Anthropic API to
draft the customer-facing reply when `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
is set in the environment, and falls back to a deterministic template
otherwise. In this sandboxed evaluation run, **0 of 35 requests** used the
LLM path (`llm_reply_path_used_pct: 0.0` in `data/eval_results.json`) —
because no key was reachable here, not because the code doesn't exist. Set
either key before running `src/evaluate.py` on your own machine to
genuinely exercise it.

## Why a mock API, not a live third-party one

The tools call a real HTTP API — `src/external_service.py` — rather than a
plain Python function, so "API integration" here means actual network
requests with actual timeouts, 404s, and validation errors to handle, not
a function call dressed up as one. It's a mock rather than a real vendor
(Shopify, Zendesk, a CRM) because this sandbox has no outbound route to
arbitrary third-party hosts. The request/response shapes mirror a typical
order-management API, and `src/tools.py` only needs a new base URL and auth
headers to point at a real one.

## Measured results

Full run in [`data/eval_results.json`](data/eval_results.json), from
`data/test_requests.json`'s 35 hand-labeled requests (ground truth is what
a correct system *should* do, not graded against this build's own rules).

- **Intent classification accuracy: 82.9% (29/35).**
- **Escalation-decision accuracy: 80.0% (28/35)** — did the agent correctly
  decide whether a human needs to be involved?
- **20 of 35 (57.1%) auto-resolved** with no human involvement; **15 of 35
  escalated.**
- **3 of 3 (100%) genuinely unclassifiable/off-topic requests were
  correctly escalated** — the safety net catches what it should.
- **0 unhandled errors across all 35 requests**, including 3 with invalid
  order IDs — every failure mode returns a structured result.
- **Latency: 38.8ms average, 104.2ms p95** for the full five-node graph
  per request (includes a real HTTP round-trip to the mock API).

### Honest limitations

The keyword/pattern rules in `src/planner.py` miss phrasings they weren't
written for. The measured misses, specifically:

- **Keyword overlap misroutes generic policy questions.** "How long do
  refunds take to process?" and "Do you offer refunds for damaged items?"
  both contain the word "refund" and get classified as `refund_request`
  instead of `faq_question`, because refund-intent keywords are checked
  before FAQ keywords. Same issue for "How do I track my order?" (contains
  "track my order", an order-status phrase) and "Can I change my shipping
  address after ordering?" (no FAQ keyword was written for this phrasing at
  all). This accounts for 4 of the 6 intent misses — distinguishing "a
  question about the policy" from "a request to act on it" is exactly the
  kind of ambiguity a keyword list struggles with and an LLM-driven planner
  would likely handle better.
- **Indirect phrasing without a keyword or a question mark.** "I've been
  waiting forever, ORD-1003 better have shipped by now" and "ORD-1008
  stopped working after two days, I want to send it back" don't contain any
  of the literal trigger phrases the rules check for, so both fall through
  to `unknown`. (The second one still ends up correctly escalated, since
  its refund amount is over the auto-approval threshold either way — the
  escalation decision is right even though the intent label isn't.)
- **Address extraction requires the literal word "to".** "My order ORD-1003
  needs a new address: 22 Baker Street, Seattle, WA" and "Please redirect my
  order ORD-1005 — new place is 5 Pine Ave, Chicago, IL" both state a clear
  new address without the word "to" before it, so `extract_new_address`
  finds nothing and the request is escalated for clarification rather than
  auto-resolved (2 of the 7 escalation misses; the other 4 are the
  keyword-overlap FAQ cases above, plus the "waiting forever" case).
- One round of tuning happened during testing, not left for the README to
  gloss over: the address-intent rule originally required exact phrases
  like "update my address," and missed "update my *shipping* address" and
  "update address for..." (see `src/planner.py`'s comment above
  `_ADDRESS_VERB_RE`). That was broadened to a verb+noun co-occurrence rule
  after evaluate.py caught it — the same kind of fix as the
  `multi-agent-financial-qa` Planner needing "for" added to its connector
  list. The limitations above are what's left after that pass, not what
  was found before it.

## Architecture

```
request_text
   |
   v
reasoning_planning  --> classify intent, extract order id / amount / address (rule-based, see above)
   |
   v
knowledge_retrieval --> order lookup (GET, read-only) or FAQ BM25 search
   |
   v
tools_apis          --> business rules decide auto-resolve vs. escalate;
   |                    real HTTP call to external_service for refunds/address changes
   v
business_action     --> where a durable audit-log write would happen
   |
   v
result_human_review --> draft the reply (LLM if a key is set, else template)
   |
   v
app.py (FastAPI /automate endpoint)
```

## Running it

Two terminals:

```bash
pip install -r requirements.txt
uvicorn src.external_service:app --port 8002       # the mock order-management API
uvicorn src.app:app --reload --port 8001            # the agent itself
```

```bash
curl -X POST http://localhost:8001/automate \
     -H "Content-Type: application/json" \
     -d '{"request_text": "Can I get a refund for ORD-1001? It arrived damaged."}'
```

Re-run the evaluation (starts its own isolated copy of the mock API, so it
never mutates `data/orders.json`):

```bash
python -m src.evaluate
```

## Stack

Python, LangGraph, FastAPI, httpx, rank_bm25 (FAQ retrieval), pydantic. No
required LLM API dependency — see "Why the Planner is rule-based" above for
the optional, genuinely-implemented OpenAI/Anthropic reply-drafting path.

## Data

`data/orders.json` and `data/faq_knowledge.json` are synthetic reference
data built for this demo, not real customer records.
`data/test_requests.json` is the hand-labeled evaluation set;
`data/eval_results.json` is that evaluation's actual output.
