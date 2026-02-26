"""Phase3 v2 runtime package.

Patch0 introduces a clean Phase3 worker entrypoint to support a redesign aligned with:
- per-message real-time predictions (lookback/rolling)
- correlation + feature-importance computed at window end (task/batch/workstation)

Patch0 keeps the worker as a no-op drain with observability hooks.
"""
