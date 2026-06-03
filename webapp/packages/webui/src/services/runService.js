// webapp/packages/webui/src/services/runService.js
//
// ISSUE-005/006 — thin client for the run registry endpoints (ISSUE-003).
// All requests inherit credentials/cookie behavior from fetchInterceptor.

import appConfig from '../config';

const BASE = (appConfig?.api?.baseUrl || '').replace(/\/$/, '');

async function jsonOrThrow(resp, action) {
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${action} failed: ${resp.status} ${text}`);
  }
  return resp.json();
}

/** POST /agents/run-code/start — returns {runId, status}. */
export async function startRun(payload) {
  const resp = await fetch(`${BASE}/agents/run-code/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(resp, 'startRun');
}

/** GET /runs — returns {runs: [{runId, agentName, status, startedAt, completedAt}, ...]}. */
export async function listRuns() {
  const resp = await fetch(`${BASE}/runs`);
  return jsonOrThrow(resp, 'listRuns');
}

/** GET /runs/{runId} — full record. */
export async function getRun(runId) {
  const resp = await fetch(`${BASE}/runs/${encodeURIComponent(runId)}`);
  return jsonOrThrow(resp, 'getRun');
}

/** POST /runs/{runId}/stop — see ISSUE-007. Returns 202 on success. */
export async function stopRun(runId) {
  const resp = await fetch(`${BASE}/runs/${encodeURIComponent(runId)}/stop`, {
    method: 'POST',
  });
  if (resp.status === 202 || resp.ok) return true;
  const text = await resp.text().catch(() => '');
  throw new Error(`stopRun failed: ${resp.status} ${text}`);
}

export default { startRun, listRuns, getRun, stopRun };
