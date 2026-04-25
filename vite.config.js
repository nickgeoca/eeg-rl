import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 8080, host: '127.0.0.1', watch: { ignored: ['**/.venv/**'] } },
});
