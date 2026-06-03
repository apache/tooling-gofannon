// ISSUE-005 — useRun subscribes via EventSource and surfaces events + status.
import { renderHook, act } from '@testing-library/react';
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import useRun from './useRun';

class FakeEventSource {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.listeners = {};
    this.onerror = null;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).concat(fn);
  }
  _emit(type, data) {
    (this.listeners[type] || []).forEach((fn) => fn({ data: JSON.stringify(data) }));
  }
  close() { this.readyState = 2; }
}

describe('useRun', () => {
  let originalES;
  beforeEach(() => {
    originalES = global.EventSource;
    global.EventSource = FakeEventSource;
    FakeEventSource.instances = [];
    // jsdom needs the CLOSED constant on the class
    global.EventSource.CLOSED = 2;
  });
  afterEach(() => {
    global.EventSource = originalES;
  });

  it('returns idle when runId is falsy', () => {
    const { result } = renderHook(() => useRun(null));
    expect(result.current.status).toBe('idle');
  });

  it('opens an EventSource for the runId and transitions to running on run_id event', () => {
    const { result } = renderHook(() => useRun('r-1'));
    const es = FakeEventSource.instances[0];
    expect(es.url).toContain('/runs/r-1/stream');
    expect(result.current.status).toBe('connecting');

    act(() => es._emit('run_id', { runId: 'r-1' }));
    expect(result.current.status).toBe('running');
  });

  it('accumulates trace events', () => {
    const { result } = renderHook(() => useRun('r-1'));
    const es = FakeEventSource.instances[0];
    act(() => es._emit('trace', { type: 'agent_start', ts: 1 }));
    act(() => es._emit('trace', { type: 'agent_end', ts: 2 }));
    expect(result.current.events).toHaveLength(2);
  });

  it('captures final payload and status on done', () => {
    const { result } = renderHook(() => useRun('r-1'));
    const es = FakeEventSource.instances[0];
    act(() => es._emit('done', {
      outcome: 'success',
      result: { ok: true },
      error: null,
      schemaWarnings: null,
      opsLog: null,
    }));
    expect(result.current.status).toBe('success');
    expect(result.current.final).toEqual({
      result: { ok: true },
      error: null,
      schemaWarnings: null,
      opsLog: null,
    });
  });

  it('closes the source on unmount', () => {
    const { unmount } = renderHook(() => useRun('r-1'));
    const es = FakeEventSource.instances[0];
    unmount();
    expect(es.readyState).toBe(2);
  });
});
