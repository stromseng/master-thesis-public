// Usage:
//   bun evals/shititong/en_text_open_ended.ts
//   EVAL_MODEL="moonshotai/Kimi-K2.5" bun evals/shititong/en_text_open_ended.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8001 bun evals/shititong/en_text_open_ended.ts
import { Effect } from "effect";
import { getOrCreateOpenEndedDataset } from "../open-ended-dataset";
import {
  evaluatorTaskLayer,
  experimentOnlyTaskLayer,
  makeOpenEndedEvaluateOnlyProgram,
  makeOpenEndedExperimentOnlyProgram,
  makeOpenEndedProgram,
  taskLayer,
} from "../open-ended-basic";

export const openEndedDatasetDefinition = {
  dataset: "shititong-en-text-open-ended",
  name: "shititong-en-text-open-ended",
  description: "Shititong English text-only maritime exam questions rewritten as open-ended evals",
  fileSegments: ["open_ended", "shititong-en-text-2026-04-10T09-12-07Z.open-ended.eval.json"],
} as const;

const getOrCreateDataset = getOrCreateOpenEndedDataset(openEndedDatasetDefinition);

export const program = makeOpenEndedProgram({
  getOrCreateDataset,
  runModelName: "question.shititong_en.openEnded.runModel",
});
export const experimentOnlyProgram = makeOpenEndedExperimentOnlyProgram({
  getOrCreateDataset,
  runModelName: "question.shititong_en.openEnded.runModel",
});

export const run = Effect.scoped(program.pipe(Effect.provide(taskLayer)));
export const runExperimentOnly = Effect.scoped(
  experimentOnlyProgram.pipe(Effect.provide(experimentOnlyTaskLayer)),
);
export const runEvaluatorsOnly = (experimentId: string) =>
  Effect.scoped(
    makeOpenEndedEvaluateOnlyProgram(experimentId).pipe(Effect.provide(evaluatorTaskLayer)),
  );

if (import.meta.main) {
  Effect.runPromise(run);
}
