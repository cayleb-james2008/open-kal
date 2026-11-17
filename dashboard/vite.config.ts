import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  // tokens.css lives in ../references (the skill folder) and is imported by src/index.css
  server: { fs: { allow: [path.resolve(import.meta.dirname, "..")] } },
});
