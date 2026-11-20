import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 接続先バックエンドの切り替え:
//   Express版 (llama.cpp)  … 既定 http://localhost:3001
//   Python版 (Transformers) … VITE_API_PROXY=http://localhost:3002 npm run dev
const apiProxy = process.env.VITE_API_PROXY ?? "http://localhost:3001";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": apiProxy,
    },
  },
});
