import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { spawnSync } from "node:child_process";
import * as JSONSchema from "effect/JSONSchema";
import { repoPath } from "../src/utils/repo";
import { EvalQuestionGroups } from "../evals/question-schema";

const pythonProjectDir = repoPath("code", "python");
const schemaOutputPath = repoPath(
  "code",
  "python",
  "src",
  "thesis",
  "generated",
  "eval_question.schema.json",
);
const modelOutputPath = repoPath(
  "code",
  "python",
  "src",
  "thesis",
  "generated",
  "eval_question_models.py",
);

mkdirSync(dirname(schemaOutputPath), { recursive: true });

const schema = JSONSchema.make(EvalQuestionGroups);
writeFileSync(schemaOutputPath, JSON.stringify(schema, null, 2));
console.log(`Wrote JSON Schema: ${schemaOutputPath}`);

const codegenArgs = [
  "--input",
  schemaOutputPath,
  "--input-file-type",
  "jsonschema",
  "--output",
  modelOutputPath,
  "--output-model-type",
  "pydantic_v2.BaseModel",
];

const uvxResult = spawnSync(
  "uvx",
  ["--from", "datamodel-code-generator", "datamodel-codegen", ...codegenArgs],
  {
    stdio: "inherit",
  },
);

if (uvxResult.status === 0) {
  console.log(`Wrote Pydantic models: ${modelOutputPath}`);
  process.exit(0);
}

console.log("uvx datamodel-codegen failed; falling back to uv run --project code/python ...");

const uvResult = spawnSync(
  "uv",
  ["run", "--project", pythonProjectDir, "datamodel-codegen", ...codegenArgs],
  {
    stdio: "inherit",
  },
);

if (uvResult.status !== 0) {
  process.exit(uvResult.status ?? 1);
}

console.log(`Wrote Pydantic models: ${modelOutputPath}`);
