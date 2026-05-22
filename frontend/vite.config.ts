import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// dev: Vite (:5173) → /api/* 를 Flask (:8050) 으로 프록시.
// prod: `npm run build` 결과인 dist/ 를 Flask 가 static serve.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://localhost:8050',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
});
