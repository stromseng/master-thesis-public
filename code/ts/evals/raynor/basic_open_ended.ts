// Usage:
//   bun evals/raynor/basic_open_ended.ts
//   EVAL_MODEL="moonshotai/Kimi-K2.5" bun evals/raynor/basic_open_ended.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8001 bun evals/raynor/basic_open_ended.ts
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
  dataset: "raynor-v2-open-ended",
  name: "raynor-v2-open-ended",
  description: "Raynor maritime navigation rules questions rewritten as open-ended evals",
  fileSegments: ["open_ended", "raynor-v2-2026-04-10T09-12-07Z.open-ended.eval.json"],
} as const;

const getOrCreateDataset = getOrCreateOpenEndedDataset(openEndedDatasetDefinition);

export const program = makeOpenEndedProgram({
  getOrCreateDataset,
  runModelName: "question.raynor.openEnded.runModel",
});
export const experimentOnlyProgram = makeOpenEndedExperimentOnlyProgram({
  getOrCreateDataset,
  runModelName: "question.raynor.openEnded.runModel",
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
