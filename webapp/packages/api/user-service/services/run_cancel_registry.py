"""Per-run cancel-token registry (ISSUE-007).

A small dict keyed by run_id that holds the active CancelToken for each
in-flight run. Maintained alongside (and independently of) the
RunRegistry from ISSUE-003 so this patch is self-contained on main:
ISSUE-007 can land before, after, or simultaneously with ISSUE-003.
When both are present, the run launcher binds a fresh CancelToken,
publishes it here so ``POST /runs/{id}/stop`` can find it, and resets
on completion.

If ISSUE-003 is NOT yet present, the stop endpoint still installs but
the agent runner (which lives in ISSUE-003's start endpoint) won't
publish tokens, so stop calls will 404. This is intentional — the
plumbing is the cheaper of the two patches and shouldn't block.
"""
from __future__ import annotations

from typing import Dict, Optional

from services.cancel_token import CancelToken


_tokens: Dict[str, CancelToken] = {}


def publish(run_id: str, token: CancelToken) -> None:
    _tokens[run_id] = token


def get(run_id: str) -> Optional[CancelToken]:
    return _tokens.get(run_id)


def clear(run_id: str) -> None:
    _tokens.pop(run_id, None)


def reset_for_tests() -> None:
    _tokens.clear()
