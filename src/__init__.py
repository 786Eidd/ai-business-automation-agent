"""AI Business Automation Agent — package root. Empty on purpose; this file
just makes `src` an explicit package so relative imports between the agent's
modules (planner, retriever, tools, reply_generator, graph, app) resolve the
same way whether you run `uvicorn src.app:app` or `python -m src.evaluate`.
"""
