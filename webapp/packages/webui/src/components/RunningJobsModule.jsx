// webapp/packages/webui/src/components/RunningJobsModule.jsx
//
// ISSUE-006 — cross-agent view of "what's running" on the home page.
// Polls GET /runs every 5s and renders each non-terminal run plus the
// most recent N completed runs. Status chip color encodes terminal
// state.
//
// Deliberately uses polling rather than per-row SSE — this is an
// at-a-glance overview, and 30+ concurrent EventSources for users
// with many agents would be wasteful.

import React, { useEffect, useState } from 'react';
import { Paper, Box, Typography, Chip, CircularProgress, Stack, Link, Button } from '@mui/material';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import PendingActionsIcon from '@mui/icons-material/PendingActions';
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
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const data = await listRuns();
        if (cancelled) return;
        // Defensive: data may be null/undefined if the endpoint returned
        // something we couldn't parse (e.g. vite SPA fallback returning
        // index.html when the api is briefly unreachable). Treat as empty
        // rather than surfacing a parse error -- this widget is
        // at-a-glance and empty is the right signal when nothing's there.
        setRuns((data && data.runs) || []);
      } catch (e) {
        if (!cancelled) {
          console.warn('RunningJobsModule: unable to fetch /runs:', e.message || e);
          setRuns([]);
        }
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

  const running = runs.filter((r) => r.status === 'running');
  const recent  = runs.filter((r) => r.status !== 'running').slice(0, MAX_COMPLETED_SHOWN);

  return (
    <Paper sx={{ overflow: 'hidden' }}>
      {/* Header matches the Data Stores / Agents pattern on the home
          page: icon + title + count chip on the left, View all on
          the right. */}
      <Box sx={{
        p: 2,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid #e4e4e7',
        bgcolor: '#fafafa',
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PendingActionsIcon sx={{ fontSize: 20, color: 'text.secondary' }} />
          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1rem' }}>Running Jobs</Typography>
          <Chip label={`All (${runs.length})`} size="small" sx={{ height: 22, fontSize: '0.72rem' }} />
          {loading && <CircularProgress size={14} sx={{ ml: 1 }} />}
        </Box>
        <Button
          variant="outlined"
          size="small"
          onClick={() => navigate('/runs')}
          disabled={runs.length === 0}
        >
          View all
        </Button>
      </Box>

      <Box sx={{ p: 2 }}>
        {!loading && running.length === 0 && recent.length === 0 && (
          <Box sx={{ py: 3, textAlign: 'center' }}>
            <Typography color="text.secondary" variant="body2">
              No runs yet
            </Typography>
            <Typography color="text.secondary" variant="caption" sx={{ display: 'block', mt: 0.5 }}>
              Kick off an agent to see live status here.
            </Typography>
          </Box>
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
      </Box>
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
