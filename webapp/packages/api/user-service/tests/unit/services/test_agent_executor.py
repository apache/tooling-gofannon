"""Tests for the Path A thread-based agent executor.

The three invariants we care about:

1. **Worker loop isolation.** Sync-blocking work inside the agent
   (think ``time.sleep`` or a sync HTTP call to CouchDB) does NOT
   stop the worker's event loop from progressing other coroutines.
2. **Trace bridge.** Events the agent appends from its side-thread
   end up on the queue owned by the worker loop, in the right order.
3. **Cross-thread cancellation.** A cancel-bit flipped from the
   worker loop is observed by the agent on its next
   ``check_should_stop()`` call.

These tests use ``time.sleep`` rather than a real HTTP call to keep
the suite hermetic; the semantics are the same — both block the
calling thread without yielding to its event loop.
"""
import asyncio
import time

import pytest

from services.agent_executor import execute_in_thread
from services.agent_trace import Trace
from services.cancel_token import CancelToken, check_should_stop, AgentStopped


@pytest.mark.asyncio
async def test_worker_loop_stays_responsive_during_agent_block():
    """A 500ms sync block in the agent must not freeze the worker
    loop for 500ms — heartbeats must keep ticking."""

    async def agent_coro():
        time.sleep(0.5)
        return "agent-done"

    heartbeats = []

    async def heartbeat():
        for _ in range(15):
            heartbeats.append(time.monotonic())
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    result = await execute_in_thread(agent_coro, CancelToken())
    await hb

    assert result == "agent-done"
    gaps_ms = [(heartbeats[i + 1] - heartbeats[i]) * 1000 for i in range(len(heartbeats) - 1)]
    # If isolation works, all gaps are ~50ms. Without it, one gap
    # is ~500ms (the sleep) and others ~0ms. Generous 200ms ceiling
    # absorbs CI jitter.
    assert max(gaps_ms) < 200, f"worker loop blocked; max gap was {max(gaps_ms):.0f}ms"


@pytest.mark.asyncio
async def test_trace_events_bridge_from_agent_thread():
    """Events appended from inside the agent reach the worker
    queue intact and in order, via call_soon_threadsafe."""
    trace = Trace()
    queue = asyncio.Queue()
    trace.attach_queue(queue, loop=asyncio.get_running_loop())

    async def agent_coro():
        for i in range(5):
            trace.append({"type": "log", "message": f"event {i}"})
        return "ok"

    await execute_in_thread(agent_coro, CancelToken())

    # Drain whatever the threadsafe routing has put on the queue.
    # call_soon_threadsafe is asynchronous-ish; give the loop a
    # tick or two to drain pending callbacks.
    await asyncio.sleep(0.01)
    received = []
    while not queue.empty():
        received.append(queue.get_nowait())

    assert len(received) == 5
    for i, ev in enumerate(received):
        assert ev["message"] == f"event {i}"


@pytest.mark.asyncio
async def test_cancel_token_works_across_threads():
    """A cancel bit flipped from the worker loop is observed by the
    agent's next check_should_stop() inside its thread."""
    cancel = CancelToken()

    async def agent_coro():
        for _ in range(100):
            check_should_stop()  # raises AgentStopped once cancel is set
            time.sleep(0.01)
        return "not-reached"

    async def stopper():
        await asyncio.sleep(0.05)
        cancel.request_stop()

    asyncio.create_task(stopper())

    with pytest.raises(AgentStopped):
        await execute_in_thread(agent_coro, cancel)
