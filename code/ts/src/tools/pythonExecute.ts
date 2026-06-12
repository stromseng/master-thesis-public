import { jsonSchema, tool } from "ai";
import { Monty, MontyError, MontySyntaxError, MontyTypingError } from "@pydantic/monty";
import type { ResourceLimits } from "@pydantic/monty";
import * as JSONSchema from "effect/JSONSchema";
import * as S from "effect/Schema";

// --- Types ---

export type PythonExecuteResourceLimits = ResourceLimits;

export type PythonExecuteResult = {
  success: boolean;
  output?: unknown;
  error?: string;
  errorType?: string;
  durationMs: number;
};

export type PythonExecuteToolOptions = {
  description?: string;
  limits?: PythonExecuteResourceLimits;
};

// --- Constants ---

export const DEFAULT_RESOURCE_LIMITS: PythonExecuteResourceLimits = {
  maxDurationSecs: 5,
  maxMemory: 10 * 1024 * 1024, // 10 MB
  maxAllocations: 100_000,
  maxRecursionDepth: 200,
};

// --- Input schema (Effect Schema → JSON Schema → Vercel AI SDK) ---

const PythonExecuteInput = S.Struct({
  code: S.String.annotations({ description: "Python code to execute" }),
});

type PythonExecuteInputType = {
  code: string;
};

const pythonExecuteInputJsonSchema = jsonSchema<PythonExecuteInputType>(
  JSONSchema.make(PythonExecuteInput),
);

// --- Core execution function ---

export function executePython(
  code: string,
  limits?: PythonExecuteResourceLimits,
): PythonExecuteResult {
  const effectiveLimits = { ...DEFAULT_RESOURCE_LIMITS, ...limits };
  const start = performance.now();

  try {
    const monty = new Monty(code);
    const result = monty.run({ limits: effectiveLimits });
    const durationMs = performance.now() - start;

    return {
      success: true,
      output: result,
      durationMs,
    };
  } catch (error) {
    const durationMs = performance.now() - start;

    if (error instanceof MontySyntaxError) {
      return {
        success: false,
        error: error.display("type-msg"),
        errorType: "SyntaxError",
        durationMs,
      };
    }

    if (error instanceof MontyTypingError) {
      return {
        success: false,
        error: error.display("type-msg"),
        errorType: "TypingError",
        durationMs,
      };
    }

    if (error instanceof MontyError) {
      return {
        success: false,
        error: error.display("type-msg"),
        errorType: error.exception.typeName,
        durationMs,
      };
    }

    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
      errorType: "UnknownError",
      durationMs,
    };
  }
}

// --- Tool factory ---

const DEFAULT_DESCRIPTION =
  "Execute Python code in a sandboxed interpreter. Use for calculations, data transformations, or algorithmic reasoning. Returns the value of the last expression.";

export function makePythonExecuteTool(options?: PythonExecuteToolOptions) {
  return tool({
    description: options?.description ?? DEFAULT_DESCRIPTION,
    inputSchema: pythonExecuteInputJsonSchema,
    execute: async ({ code }) => executePython(code, options?.limits),
  });
}

// --- Default tool instance ---

export const pythonExecute = makePythonExecuteTool();
