import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts", "evals/**/*.test.ts"],
    env: {
      ...loadEnv("", process.cwd(), ""),
      ...loadEnv("", "../..", ""),
    },
    setupFiles: ["tests/support/vitest.setup.ts"],
  },
});
