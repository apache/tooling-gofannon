"""Cooperative cancellation primitive (ISSUE-007).

A ``CancelToken`` is a tiny flag-with-a-method-name. It's bound to an
asyncio task via a ``ContextVar`` so structurally distant code paths
(LLM service, data-store proxy, gofannon-client) can poll it without a
signature change. asyncio task context inheritance means a chained
agent automatically shares the parent's token — stopping the parent
stops the children.

Enforcement happens at *structural boundaries*: at entry to each
tool invocation we call ``check_should_stop()``, which raises
``AgentStopped`` if the token has been set. In-flight LLM HTTP
requests finish naturally; only the *next* observable action is
interrupted. This is intentionally less aggressive than
``task.cancel()``, which would interrupt mid-await and skip cleanup
handlers.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


class AgentStopped(Exception):
    """Raised when a run's cancel token is set and a structural check fires."""


@dataclass
class CancelToken:
    """Trivial boolean flag, intentionally not thread-safe.

    asyncio task switching never preempts a sync attribute write, so the
    obvious dataclass works. Adding a Lock would be slower without
    being safer.
    """
    _stopped: bool = False

    def request_stop(self) -> None:
        self._stopped = True

    def is_stopped(self) -> bool:
        return self._stopped


_token: ContextVar[Optional[CancelToken]] = ContextVar("cancel_token", default=None)


def bind_token(token: CancelToken):
    """Set the active token for the current task. Returns the contextvar
    Token so the caller can ``reset()`` if they need to."""
    return _token.set(token)


def reset_token(ctx_token) -> None:
    _token.reset(ctx_token)


def current_token() -> Optional[CancelToken]:
    return _token.get()


def should_stop() -> bool:
    tok = _token.get()
    return tok.is_stopped() if tok else False


def check_should_stop() -> None:
    """Raise ``AgentStopped`` if the bound token has been set.

    Call at the entry to each tool function so a stop request takes
    effect at the next structural boundary.
    """
    if should_stop():
        raise AgentStopped("Run was stopped by user")
