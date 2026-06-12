import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

let cachedRoot: string | undefined;

/**
 * Walk up from `start` until a directory with pyproject.toml, code/, and data/ is found.
 * The result is cached after the first successful lookup.
 */
export function findRepoRoot(start?: string): string {
  if (cachedRoot !== undefined) return cachedRoot;

  const origin = start ?? dirname(fileURLToPath(import.meta.url));
  let current = resolve(origin);

  while (true) {
    if (
      existsSync(join(current, "pyproject.toml")) &&
      existsSync(join(current, "code")) &&
      existsSync(join(current, "data"))
    ) {
      cachedRoot = current;
      return current;
    }

    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }

  throw new Error(`Could not locate repo root from ${origin}`);
}

export const REPO_ROOT = findRepoRoot();

/** Resolve a path relative to the repo root. */
export const repoPath = (...segments: string[]) => join(REPO_ROOT, ...segments);

/** Resolve a path under data/. */
export const dataPath = (...segments: string[]) => repoPath("data", ...segments);

/** Resolve a path under data/evals/. */
export const evalDataPath = (...segments: string[]) => dataPath("evals", ...segments);
