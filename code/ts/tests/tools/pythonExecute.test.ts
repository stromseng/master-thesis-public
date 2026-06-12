import { describe, expect, it } from "@effect/vitest";
import {
  executePython,
  makePythonExecuteTool,
  DEFAULT_RESOURCE_LIMITS,
} from "../../src/tools/pythonExecute";

describe("executePython", () => {
  it("evaluates simple arithmetic", () => {
    const result = executePython("1 + 2");
    expect(result.success).toBe(true);
    expect(result.output).toBe(3);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("evaluates exponentiation", () => {
    const result = executePython("2 ** 10");
    expect(result.success).toBe(true);
    expect(result.output).toBe(1024);
  });

  it("handles function definitions and calls", () => {
    const result = executePython(`
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

fibonacci(10)
`);
    expect(result.success).toBe(true);
    expect(result.output).toBe(55);
  });

  it("handles list comprehensions", () => {
    const result = executePython("[x * x for x in range(5)]");
    expect(result.success).toBe(true);
    expect(result.output).toEqual([0, 1, 4, 9, 16]);
  });

  it("handles string operations", () => {
    const result = executePython("'hello' + ' ' + 'world'");
    expect(result.success).toBe(true);
    expect(result.output).toBe("hello world");
  });

  it("returns SyntaxError for invalid syntax", () => {
    const result = executePython("def foo(");
    expect(result.success).toBe(false);
    expect(result.errorType).toBe("SyntaxError");
    expect(result.error).toBeDefined();
  });

  it("returns RuntimeError for division by zero", () => {
    const result = executePython("1 / 0");
    expect(result.success).toBe(false);
    expect(result.errorType).toBe("ZeroDivisionError");
    expect(result.error).toBeDefined();
  });

  it("returns error for undefined variable", () => {
    const result = executePython("undefined_var");
    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });

  it("respects custom resource limits", () => {
    const result = executePython("1 + 1", {
      maxDurationSecs: 1,
      maxMemory: 1024 * 1024,
      maxAllocations: 10_000,
      maxRecursionDepth: 50,
    });
    expect(result.success).toBe(true);
    expect(result.output).toBe(2);
  });
});

describe("makePythonExecuteTool", () => {
  it("creates a tool with custom limits", () => {
    const customTool = makePythonExecuteTool({
      limits: { maxDurationSecs: 1 },
    });
    expect(customTool).toBeDefined();
  });

  it("creates a tool with custom description", () => {
    const customTool = makePythonExecuteTool({
      description: "Custom python tool",
    });
    expect(customTool).toBeDefined();
  });
});

describe("DEFAULT_RESOURCE_LIMITS", () => {
  it("has expected default values", () => {
    expect(DEFAULT_RESOURCE_LIMITS.maxDurationSecs).toBe(5);
    expect(DEFAULT_RESOURCE_LIMITS.maxMemory).toBe(10 * 1024 * 1024);
    expect(DEFAULT_RESOURCE_LIMITS.maxAllocations).toBe(100_000);
    expect(DEFAULT_RESOURCE_LIMITS.maxRecursionDepth).toBe(200);
  });
});
