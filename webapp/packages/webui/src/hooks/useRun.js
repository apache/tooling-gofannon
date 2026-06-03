// webapp/packages/webui/src/hooks/useRun.js
//
// ISSUE-005 — subscribe to a run's SSE stream with replay-then-live
// semantics. Backed by GET /runs/{run_id}/stream from ISSUE-003's run
// registry.
//
// The hook OPENS the stream on mount and CLOSES it on unmount. Closing
// does not affect the underlying run (that's the whole point of the
// registry). Reopening reconnects and replays.

import { useEffect, useRef, useState } from 'react';

export default function useRun(runId) {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState('connecting');  // connecting|running|success|error|stopped
  const [final, setFinal] = useState(null);            // {result, error, schemaWarnings, opsLog}
  const sourceRef = useRef(null);

  useEffect(() => {
    if (!runId) {
      setStatus('idle');
      return;
    }
    setEvents([]);
    setFinal(null);
    setStatus('connecting');

    const url = `/runs/${encodeURIComponent(runId)}/stream`;
    const es = new EventSource(url, { withCredentials: true });
    sourceRef.current = es;

    es.addEventListener('run_id', () => {
      setStatus((s) => (s === 'connecting' ? 'running' : s));
    });

    es.addEventListener('trace', (e) => {
      try {
        const ev = JSON.parse(e.data);
        setEvents((prev) => [...prev, ev]);
      } catch {
        // Drop unparseable frames; the run continues independent of UI.
      }
    });

    es.addEventListener('done', (e) => {
      try {
        const payload = JSON.parse(e.data);
        setFinal({
          result: payload.result,
          error: payload.error,
          schemaWarnings: payload.schemaWarnings,
          opsLog: payload.opsLog,
        });
        setStatus(payload.outcome || 'success');
      } catch {
        setStatus('error');
      } finally {
        es.close();
        sourceRef.current = null;
      }
    });

    es.onerror = () => {
      // EventSource auto-reconnects on transient errors; only flip status
      // if the connection terminated for good (readyState CLOSED).
      if (es.readyState === EventSource.CLOSED) {
        setStatus((s) => (s === 'running' || s === 'connecting' ? 'error' : s));
      }
    };

    return () => {
      es.close();
      sourceRef.current = null;
    };
  }, [runId]);

  return { events, status, final };
}
