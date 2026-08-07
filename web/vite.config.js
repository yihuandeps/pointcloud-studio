import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    open: false,
    proxy: {
      // 高精度模式：转发到 server/app.py 的 FastAPI（阶段二）
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
    // 如需 onnxruntime-web 的多线程 WASM（SharedArrayBuffer），解开下面两行。
    // 注意：开启后跨域资源需带 CORP 头，HF 镜像可能被拦，出问题就再注释掉。
    // headers: {
    //   'Cross-Origin-Opener-Policy': 'same-origin',
    //   'Cross-Origin-Embedder-Policy': 'credentialless',
    // },
  },

  // transformers.js 自带预打包的 onnxruntime-web，交给 Vite 预构建会出问题
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },

  worker: {
    format: 'es',
  },

  build: {
    target: 'esnext', // 需要 top-level await
    chunkSizeWarningLimit: 2048,
  },
});
