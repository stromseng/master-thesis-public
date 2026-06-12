import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { embedMany } from "ai";

const apiKey = process.env.LITE_LLM_API_KEY;
if (!apiKey) {
  throw new Error("LITE_LLM_API_KEY environment variable is required");
}

// Create a custom provider for NTNU's LiteLLM gateway
const litellmProvider = createOpenAICompatible({
  name: "litellm",
  apiKey,
  baseURL: "https://llm.hpc.ntnu.no/v1",
});

const result = await embedMany({
  model: litellmProvider.embeddingModel("Qwen/Qwen3-Embedding-8B"),
  values: ["hello world", "this is another sentence"],
});

console.log("Embeddings:");
for (const [i, embedding] of result.embeddings.entries()) {
  console.log(
    `  [${i}]: ${embedding.length} dimensions, first 5: [${embedding.slice(0, 5).join(", ")}...]`,
  );
}
