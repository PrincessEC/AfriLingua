import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import vike from "vike/plugin";
import tailwindcss from "@tailwindcss/vite";

const ENGINE = process.env.AFRILINGUA_ENGINE_URL ?? "http://127.0.0.1:8000";

const proxyEntry = { target: ENGINE, changeOrigin: true, ws: true } as const;

export default defineConfig({
  plugins: [vike(), react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": proxyEntry,
      "/v1": proxyEntry,
      "/mcp": proxyEntry,
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": proxyEntry,
      "/v1": proxyEntry,
      "/mcp": proxyEntry,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
  },
});
