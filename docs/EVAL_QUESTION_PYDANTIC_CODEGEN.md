# Eval Question Schema -> Pydantic Codegen

## What this is

The canonical eval question model is defined in TypeScript as an Effect Schema in:

- `code/ts/evals/question-schema.ts`

This repo includes automation to generate:

1. JSON Schema from the Effect Schema
2. Pydantic v2 models from that JSON Schema

This keeps TypeScript and Python models aligned from one source of truth.

## One command

From repo root:

```bash
bun run gen:question-pydantic
```

## What gets generated

The command writes:

- `code/python/src/thesis/generated/eval_question.schema.json`
- `code/python/src/thesis/generated/eval_question_models.py`

## How it works

The generator script is:

- `code/ts/scripts/generate-question-pydantic.ts`

Flow:

1. `effect/JSONSchema` converts `EvalQuestion` to JSON Schema.
2. `datamodel-codegen` converts JSON Schema to Pydantic v2 models.

The script first tries `uvx --from datamodel-code-generator ...` and falls back to `uv run --project code/python ...` if needed.

## Regeneration workflow

When you change the TS schema in `code/ts/evals/question-schema.ts`:

1. Re-run `bun run gen:question-pydantic`.
2. Commit the updated generated files.

## Notes

- `eval_question_models.py` is generated code and can be overwritten.
- Prefer editing `code/ts/evals/question-schema.ts` instead of manually editing generated Python models.
