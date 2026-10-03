import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server: the frozen `/api` contract is proxied to the local FastAPI
// backend (apps/vlm-jlens-demo/backend). In production the backend serves the
// built `dist/` directly, so the app keeps same-origin `/api` requests.
// If the backend listens elsewhere, change the proxy target below.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8787',
    },
  },
  build: {
    outDir: 'dist',
  },
});
