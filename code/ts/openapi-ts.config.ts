import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: "http://localhost:8001/openapi.json",
  output: {
    path: "src/generated/python-api",
  },
  plugins: ["@hey-api/typescript", "@hey-api/sdk"],
});
