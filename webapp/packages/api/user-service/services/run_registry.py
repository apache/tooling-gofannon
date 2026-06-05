"""Run registry — in-process record of agent runs (ISSUE-003).

A ``RunRecord`` holds everything the runs UX needs to display, replay, and
(in ISSUE-007) cancel an in-flight or recently-completed agent run. The
``RunRegistry`` is a singleton (per process) that owns the records.

Design constraints:

- **Single process.** Records live in memory. A multi-replica deployment
  would need sticky sessions or a shared store; that's out of scope for
  Phase 1 (issue text calls this out explicitly).
- **Closing the SSE connection does not kill the run.** Each record's
  asyncio Task runs independently of any subscriber. The registry holds
  a strong reference so the task isn't GC'd. The SSE response is a thin
  subscriber that reads events from a per-record subscriber queue.
- **Replay-then-live.** A new subscriber sees every event already in
  ``record.trace.events`` (replay), then transitions to live consumption
  via its own queue. Multiple subscribers per record are supported —
  each gets its own queue, attached to the trace via ``Trace.attach_queue``
  semantics. (Today Trace only supports one queue; this module wraps that
  by fanning out at append-time. See ``_FanoutTrace`` below.)
- **Eviction.** On completion, the record stays in memory for
  ``EVICTION_TTL_SECONDS`` (default 3600). After that, it's evicted —
  callers asking for it get 404. Persistence to CouchDB is a follow-up.
- **Auth scoping.** Records track ``user_id``. Lookup-by-id callers must
  check ownership before exposing the record. (After ISSUE-002 lands,
  widen to workspace membership.)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set

from services.agent_trace import Trace
from time_utils import naive_utc_now


EVICTION_TTL_SECONDS = 3600  # 1 hour
DONE_SENTINEL = object()


class _FanoutTrace(Trace):
    """A Trace that supports multiple attached subscriber queues.

    The base ``Trace.attach_queue`` overwrites the single queue slot. We
    need fanout for the registry: a long-lived "live" queue is consumed
    by SSE subscribers, and new subscribers also get the existing
    ``self.events`` list replayed before going live. Each subscriber's
    own queue is set via ``add_subscriber``; removal via
    ``remove_subscriber``.

    Path A integration: each subscriber tracks the loop that owns its
    queue. When append() fires from a thread other than the queue\'s
    owning loop (the executor runs the agent on a side thread), the
    put_nowait is routed through call_soon_threadsafe so the queue is
    only mutated by its owning loop. When no loop is provided
    (legacy callers), behavior is the original direct put_nowait.
    """

    def __init__(self) -> None:
        super().__init__()
        # subscriber → optional owning_loop. None means "emit from any
        # thread without bridging" (the legacy code path before
        # Path A).
        self._subscribers: Dict[asyncio.Queue, Optional[asyncio.AbstractEventLoop]] = {}

    def add_subscriber(
        self,
        queue: asyncio.Queue,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._subscribers[queue] = loop

    def remove_subscriber(self, queue: asyncio.Queue) -> None:
        self._subscribers.pop(queue, None)

    def append(self, event: Dict[str, Any]) -> None:
        # Append to the immutable history first (parent stores in
        # self.events), then fan out to subscribers.
        super().append(event)
        for q, loop in list(self._subscribers.items()):
            if loop is not None:
                # Route the put via the queue\'s owning loop. Safe
                # even when we\'re already on that loop\'s thread.
                loop.call_soon_threadsafe(self._safe_subscriber_put, q, event)
            else:
                try:
                    q.put_nowait(event)
                except Exception:
                    # Queue full or closed; drop the event for that
                    # subscriber rather than blocking other emitters.
                    pass

    @staticmethod
    def _safe_subscriber_put(queue: asyncio.Queue, event: Dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except Exception:
            pass


@dataclass
class RunRecord:
    """One agent run's full state."""

    run_id: str
    user_id: str
    agent_name: str
    # CouchDB doc id of the saved agent this run came from, when applicable.
    # None for runs initiated from the create-flow sandbox (no saved agent yet).
    # Used by the runs UI to deep-link to the per-agent runs page.
    agent_id: Optional[str] = None
    # The input dict the run was launched with. Stored so the runs UI
    # can pre-fill the form when revisiting a past run via deep link.
    # None for old records that predate this field.
    input_dict: Optional[Dict[str, Any]] = None
    started_at: datetime = field(default_factory=naive_utc_now)
    status: Literal["running", "success", "error", "stopped"] = "running"
    trace: _FanoutTrace = field(default_factory=_FanoutTrace)
    task: Optional[asyncio.Task] = None  # Set by the registry on register
    result: Optional[Any] = None
    error: Optional[str] = None
    schema_warnings: Optional[List[str]] = None
    ops_log: Optional[List[Dict[str, Any]]] = None
    completed_at: Optional[datetime] = None
    # Wall-clock timestamp (time.monotonic) for cheap eviction comparison.
    _completed_at_monotonic: Optional[float] = None

    def to_summary(self) -> Dict[str, Any]:
        """Subset suitable for list endpoints — no events, no result body.

        Does include inputDict and error so the per-agent past-runs UI
        can render an input preview and inline error message without
        firing N+1 full-record fetches for each row.
        """
        return {
            "runId": self.run_id,
            "agentId": self.agent_id,
            "agentName": self.agent_name,
            "status": self.status,
            "startedAt": self.started_at.isoformat(),
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "inputDict": self.input_dict,
            "error": self.error,
        }

    def to_full(self) -> Dict[str, Any]:
        return {
            **self.to_summary(),
            "events": list(self.trace.events),
            "result": self.result,
            "error": self.error,
            "schemaWarnings": self.schema_warnings,
            "opsLog": self.ops_log,
            # inputDict echoes the input the run was launched with so
            # the runs UI can pre-fill its form when a user revisits a
            # past run via deep link.
            "inputDict": self.input_dict,
        }


class RunRegistry:
    """Process-wide registry of RunRecords."""

    def __init__(self) -> None:
        self._records: Dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    def new_record(
        self,
        *,
        user_id: str,
        agent_name: str,
        agent_id: Optional[str] = None,
        input_dict: Optional[Dict[str, Any]] = None,
    ) -> RunRecord:
        """Create a record with a fresh run_id and register it."""
        run_id = str(uuid.uuid4())
        record = RunRecord(
            run_id=run_id,
            user_id=user_id,
            agent_name=agent_name,
            agent_id=agent_id,
            input_dict=input_dict,
        )
        self._records[run_id] = record
        return record

    def bind_task(self, record: RunRecord, task: asyncio.Task) -> None:
        record.task = task

    def get(self, run_id: str) -> Optional[RunRecord]:
        self._evict_expired()
        return self._records.get(run_id)

    def list_for_user(
        self,
        user_id: str,
        limit: int = 100,
        agent_id: Optional[str] = None,
    ) -> List[RunRecord]:
        self._evict_expired()
        out = [r for r in self._records.values() if r.user_id == user_id]
        if agent_id is not None:
            out = [r for r in out if r.agent_id == agent_id]
        out.sort(key=lambda r: r.started_at, reverse=True)
        return out[:limit]

    def mark_complete(
        self,
        record: RunRecord,
        *,
        status: Literal["success", "error", "stopped"],
        result: Any = None,
        error: Optional[str] = None,
        schema_warnings: Optional[List[str]] = None,
        ops_log: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        record.status = status
        record.result = result
        record.error = error
        record.schema_warnings = schema_warnings
        record.ops_log = ops_log
        record.completed_at = naive_utc_now()
        record._completed_at_monotonic = time.monotonic()

        # Signal end-of-stream to all current subscribers.
        for q in list(record.trace._subscribers):
            try:
                q.put_nowait(DONE_SENTINEL)
            except Exception:
                pass

    def _evict_expired(self) -> None:
        """Drop completed records whose TTL has passed.

        Cheap O(n) scan; n is small for the foreseeable future. If this
        becomes hot, add a min-heap of completed records keyed by
        ``_completed_at_monotonic``.
        """
        now = time.monotonic()
        expired = [
            run_id
            for run_id, r in self._records.items()
            if r._completed_at_monotonic is not None
            and (now - r._completed_at_monotonic) > EVICTION_TTL_SECONDS
        ]
        for run_id in expired:
            self._records.pop(run_id, None)


_registry: Optional[RunRegistry] = None


def get_run_registry() -> RunRegistry:
    """Process-singleton accessor."""
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


def reset_run_registry_for_tests() -> None:
    """Tests-only — discard the singleton."""
    global _registry
    _registry = None
