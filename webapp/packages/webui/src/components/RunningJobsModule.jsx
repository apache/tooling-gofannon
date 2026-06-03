// webapp/packages/webui/src/components/RunningJobsModule.jsx
//
// ISSUE-006 — cross-agent view of "what's running" on the home page.
// Polls GET /runs every 5s and renders each non-terminal run plus the
// most recent N completed runs. Status chip color encodes terminal state.
//
// Deliberately uses polling rather than per-row SSE — this is an
// at-a-glance overview, and 30+ concurrent EventSources for users with
// many agents would be wasteful.

import React, { useEffect, useState } from 'react';
import { Paper, Box, Typography, Chip, CircularProgress, Stack, Link } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { listRuns } from '../services/runService';

const POLL_INTERVAL_MS = 5000;
const MAX_COMPLETED_SHOWN = 5;

const STATUS_CHIP = {
  running: { label: 'running', color: 'info' },
  success: { label: 'success', color: 'success' },
  error:   { label: 'error',   color: 'error' },
  stopped: { label: 'stopped', color: 'default' },
};

export default function RunningJobsModule() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const data = await listRuns();
        if (cancelled) return;
        setRuns(data.runs || []);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e.message || String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchOnce();
    const id = setInterval(fetchOnce, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const running  = runs.filter((r) => r.status === 'running');
  const recent   = runs.filter((r) => r.status !== 'running').slice(0, MAX_COMPLETED_SHOWN);

  return (
    <Paper sx={{ p: 2, mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
        <Typography variant="h6">Running Jobs</Typography>
        {loading && <CircularProgress size={16} />}
      </Box>

      {error && (
        <Typography variant="body2" color="error" sx={{ mb: 1 }}>
          {error}
        </Typography>
      )}

      {!loading && running.length === 0 && recent.length === 0 && (
        <Typography variant="body2" color="text.secondary">
          No runs yet. Kick off an agent to see live status here.
        </Typography>
      )}

      {running.length > 0 && (
        <Box sx={{ mb: 1.5 }}>
          <Typography variant="caption" color="text.secondary">In flight</Typography>
          <Stack spacing={0.5} sx={{ mt: 0.5 }}>
            {running.map((r) => <RunRow key={r.runId} run={r} />)}
          </Stack>
        </Box>
      )}

      {recent.length > 0 && (
        <Box>
          <Typography variant="caption" color="text.secondary">Recent</Typography>
          <Stack spacing={0.5} sx={{ mt: 0.5 }}>
            {recent.map((r) => <RunRow key={r.runId} run={r} />)}
          </Stack>
        </Box>
      )}
    </Paper>
  );
}

function RunRow({ run }) {
  const cfg = STATUS_CHIP[run.status] || { label: run.status, color: 'default' };
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.875rem' }}>
      <Chip size="small" label={cfg.label} color={cfg.color} />
      <Link component={RouterLink} to={`/runs/${run.runId}`} sx={{ flex: 1, textDecoration: 'none' }}>
        {run.agentName}
      </Link>
      <Typography variant="caption" color="text.secondary">
        {formatTimeAgo(run.startedAt)}
      </Typography>
    </Box>
  );
}

function formatTimeAgo(iso) {
  if (!iso) return '';
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return `${Math.round(diff / 3600)}h ago`;
}
