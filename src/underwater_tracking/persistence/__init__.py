# src/underwater_tracking/persistence/__init__.py
"""Operational persistence: SQLite repositories plus the JSONL frame log.

SQLite holds LangGraph checkpoints, runtime events, plan versions, the
DecisionLedger, expert directives, and LLM metadata (spec 5.4); the JSONL
frame log records one visualization/replay frame every 10 seconds.
"""
