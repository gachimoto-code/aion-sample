import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// 接続先バックエンドは frontend/.env の VITE_API_PROXY で切り替える
// （.env.example 参照。未設定なら Express 版の http://localhost:3001）
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiProxy = env.VITE_API_PROXY ?? "http://localhost:3001";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": apiProxy,
      },
    },
  };
});
