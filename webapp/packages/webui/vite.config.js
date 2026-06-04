import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    cors: true, // allow any origin, TODO fix for production
    strictPort: false,
    proxy: {
      // Mirror webapp/infra/docker/nginx.conf — same top-level prefixes
      // route to the api container. dev-tail.sh runs Vite without nginx
      // in front of it, so we proxy here to keep the SPA same-origin.
      // Keep this list in sync with nginx.conf when new top-level routers
      // are mounted.
      '^/(auth|admin|agents|chat|data-store|demos|deployments|echo|health|log|mcp|providers|rest|runs|sessions|specs|users)(/|$)': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    watch: {
      ignored: [
        '**/test-results/**',
        '**/playwright-report/**',
        '**/.auth/**',
        '**/coverage/**',
        '**/htmlcov/**',
      ],
    },
  },
  
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/tests/setup.js',
  },
});
