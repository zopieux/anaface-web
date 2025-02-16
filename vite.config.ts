import { defineConfig } from 'vite'
import solid from 'vite-plugin-solid'
import wasm from 'vite-plugin-wasm'

export default defineConfig({
  plugins: [
    solid(),
    wasm(),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    fs: {
      allow: ['.', '/nix/store'],
    },
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  worker: {
    format: 'es',
    plugins: () => [wasm()],
  },
  build: {
    rollupOptions: {
      external: ['onnxruntime-web'],
    },
  },
})
