"""Thread-based agent executor (Path A).

Background: the streaming endpoint creates ``asyncio.create_task(run_agent_task())``
which runs the agent on the SAME event loop as the worker that received
the request. Inside, the user agent calls ``files_ns.set("foo", bar)``
synchronously — that\'s a blocking HTTP call to CouchDB. While that call
is in flight, the worker\'s event loop is fully blocked, no other
coroutine on that loop progresses. With many sequential sync calls
(asvs_download_repo does ~1300 of them) the worker is effectively
unavailable for the duration of an agent run, even though only one of
N workers is supposed to be busy.

Fix: hand the agent off to its own OS thread with its own asyncio event
loop. The worker\'s loop stays free; sync calls inside the agent only
block the agent thread\'s loop. Trace events bridge back to the
worker\'s queue via ``call_soon_threadsafe``; cancellation is a
thread-safe bool flip that the agent picks up on its next
``check_should_stop()`` call.

This module is intentionally thin — most of the cross-thread plumbing
lives in agent_trace.py and run_registry.py where the Trace knows how
to publish to a queue owned by another loop. The executor itself just
spins up the thread, runs the supplied factory inside a fresh loop,
and bridges the result back via a Future on the parent loop.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Callable

from services.cancel_token import CancelToken, bind_token


async def execute_in_thread(
    coro_factory: Callable[[], Awaitable[Any]],
    cancel_token: CancelToken,
    thread_name: str = "agent-executor",
) -> Any:
    """Run ``coro_factory()`` in a fresh thread with its own event loop.

    ``coro_factory`` is a zero-arg callable that returns a coroutine.
    We accept a factory rather than a coroutine because a coroutine
    object\'s loop affinity is decided when its first ``await`` runs;
    instantiating it INSIDE the target thread keeps everything bound
    to the agent thread\'s loop. Passing a pre-built coroutine here
    would attach it to the caller\'s loop and defeat the point.

    The CancelToken is re-bound inside the target thread\'s context
    so the agent\'s ``check_should_stop()`` sees it. ContextVar
    bindings don\'t cross thread boundaries on their own; we re-set
    explicitly. Setting the bit from the worker loop (POST /runs/
    <id>/stop) is fine — bools are thread-safe to read and write.

    Returns the coroutine\'s result or re-raises its exception on
    the calling task. Exceptions propagate naturally through the
    Future\'s ``set_exception``.
    """
    parent_loop = asyncio.get_running_loop()
    result_future: asyncio.Future = parent_loop.create_future()

    def thread_entry() -> None:
        thread_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(thread_loop)
            # The cancel token contextvar binding doesn\'t inherit
            # across thread boundaries; bind it inside this thread\'s
            # context so check_should_stop() at structural boundaries
            # inside the agent sees the token.
            bind_token(cancel_token)
            try:
                # Build the coroutine inside this thread so its loop
                # affinity is this thread\'s loop, not the caller\'s.
                coro = coro_factory()
                result = thread_loop.run_until_complete(coro)
                parent_loop.call_soon_threadsafe(_set_result_safe, result_future, result)
            except BaseException as exc:
                parent_loop.call_soon_threadsafe(_set_exception_safe, result_future, exc)
        finally:
            try:
                thread_loop.close()
            except Exception:
                pass

    threading.Thread(target=thread_entry, daemon=True, name=thread_name).start()
    return await result_future


def _set_result_safe(future: asyncio.Future, result: Any) -> None:
    """Set a future\'s result iff still pending.

    If the parent task was cancelled or otherwise resolved before the
    agent thread finished, the future may already be done. Silently
    skip the second resolve rather than raising InvalidStateError back
    into call_soon_threadsafe (which would surface as an uncaught
    exception on the parent loop\'s exception handler).
    """
    if not future.done():
        future.set_result(result)


def _set_exception_safe(future: asyncio.Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)
