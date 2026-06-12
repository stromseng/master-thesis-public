// Usage:
//   bun evals/pei2024/uk_open_ended_test.ts
//   EVAL_MODEL="moonshotai/Kimi-K2.5" bun evals/pei2024/uk_open_ended_test.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8001 bun evals/pei2024/uk_open_ended_test.ts
import { Effect } from "effect";
import { getOrCreateSlicedOpenEndedDataset } from "../open-ended-dataset";
import {
  evaluatorTaskLayer,
  experimentOnlyTaskLayer,
  makeOpenEndedEvaluateOnlyProgram,
  makeOpenEndedExperimentOnlyProgram,
  makeOpenEndedProgram,
  taskLayer,
} from "../open-ended-basic";

const getOrCreateDataset = getOrCreateSlicedOpenEndedDataset({
  dataset: "pei2024-uk-open-ended-test",
  name: "pei2024-uk-open-ended-test",
  description: "PEI 2024 UK theory test open-ended (10 question test subset)",
  fileSegments: ["open_ended", "pei2024-uk-2026-04-09T11-52-35Z.open-ended.eval.json"],
  maxQuestions: 10,
});

export const program = makeOpenEndedProgram({
  getOrCreateDataset,
  runModelName: "question.pei2024_uk.openEnded.test.runModel",
});
export const experimentOnlyProgram = makeOpenEndedExperimentOnlyProgram({
  getOrCreateDataset,
  runModelName: "question.pei2024_uk.openEnded.test.runModel",
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
