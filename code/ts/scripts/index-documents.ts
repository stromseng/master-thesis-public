import { Glob } from "bun";
import { basename } from "path";
import { Effect, Layer, Schema } from "effect";
import * as Progress from "effective-progress";
import { dataPath } from "../src/utils/repo";
import { Qdrant, DocumentChunkPayload } from "../src/services/Qdrant";
import { EmbeddingProvider } from "../src/services/embeddings";
import { DenseEmbedding } from "../src/services/embeddings/Dense";
import { SparseEmbedding } from "../src/services/embeddings/Sparse";
import { Chunking } from "../src/services/Chunking";
import { PythonApiClient } from "../src/services/PythonApiClient";

class IndexScriptError extends Schema.TaggedError<IndexScriptError>()("IndexScriptError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

const DATA_DIR = dataPath("rag", "processed");
const FILE_CONCURRENCY = 1;

const program = Effect.gen(function* () {
  const qdrant = yield* Qdrant;
  const chunking = yield* Chunking;

  // Find all markdown files using Bun's Glob
  const glob = new Glob("**/*.md");
  const files = Array.from(glob.scanSync(DATA_DIR));

  console.log(`Found ${files.length} markdown files in ${DATA_DIR}`);

  const processFile = Effect.fn("processFile")(function* (file: string) {
    const fullPath = `${DATA_DIR}/${file}`;
    const filename = basename(file);

    // Check if file already indexed
    const alreadyIndexed = yield* qdrant.checkFileIndexed(filename);
    if (alreadyIndexed) {
      return { filename, indexed: false, chunkCount: 0 };
    }

    const content = yield* Effect.tryPromise({
      try: () => Bun.file(fullPath).text(),
      catch: (e) => new IndexScriptError({ reason: `Failed to read ${fullPath}`, cause: e }),
    });

    // Chunk the file via Chunking service
    const chunkResults = yield* chunking
      .chunk(content)
      .pipe(
        Effect.mapError(
          (e) => new IndexScriptError({ reason: `Failed to chunk ${filename}`, cause: e }),
        ),
      );

    const chunks = chunkResults.map(
      (c) =>
        new DocumentChunkPayload({
          text: c.text,
          start_index: c.start_index,
          end_index: c.end_index,
          token_count: c.token_count,
          context: c.context ?? null,
          source: filename,
        }),
    );

    // Index with chunk embedding progress
    yield* Progress.task(
      Effect.gen(function* () {
        const progress = yield* Progress.Progress;
        const taskId = yield* Progress.Task;

        yield* qdrant.indexChunks(chunks, {
          onProgress: (processed, _total) => progress.updateTask(taskId, { succeeded: processed }),
        });
      }),
      { description: `Embedding ${filename}`.slice(0, 30), total: chunks.length },
    );

    // Mark file as indexed
    yield* qdrant.markFileIndexed(filename);

    return { filename, indexed: true, chunkCount: chunks.length };
  });

  // Process files with progress
  const results = yield* Progress.forEach(files, (file) => processFile(file), {
    description: "Indexing files",
    concurrency: FILE_CONCURRENCY,
  });

  // Summary
  const indexed = results.filter((r) => r.indexed);
  const skipped = results.filter((r) => !r.indexed);
  const totalChunks = indexed.reduce((sum, r) => sum + r.chunkCount, 0);

  if (indexed.length === 0) {
    console.log("\nNo new or changed files to index");
  } else {
    console.log(`\nIndexed ${totalChunks} chunks from ${indexed.length} files`);
    console.log(`Skipped ${skipped.length} unchanged files`);
  }
});

// Layer setup — configurable via env vars:
//   DENSE_METHOD=litellm|fastembed (default: fastembed)
//   DENSE_MODEL=<model-name>       (optional, overrides default for chosen method)
//   CHUNKING_METHOD=semantic|recursive (default: recursive)
const denseMethod = process.env.DENSE_METHOD ?? "fastembed";
const denseModel = process.env.DENSE_MODEL;

const denseLayer =
  denseMethod === "litellm"
    ? DenseEmbedding.LiteLLM(
        denseModel
          ? { modelName: denseModel, vectorName: denseModel.split("/").pop()! }
          : undefined,
      )
    : DenseEmbedding.Fastembed(
        denseModel
          ? { modelName: denseModel, vectorName: denseModel.split("/").pop()! }
          : undefined,
      );

const chunkingMethod = process.env.CHUNKING_METHOD ?? "recursive";
const chunkingLayer = chunkingMethod === "semantic" ? Chunking.Semantic() : Chunking.Recursive();

console.log(
  `Config: dense=${denseMethod}${denseModel ? ` (${denseModel})` : ""}, chunking=${chunkingMethod}`,
);

const qdrantLayer = Qdrant.skyhigh;

const embeddingProviderLayer = EmbeddingProvider.make(denseLayer, SparseEmbedding.Default);

const qdrantWithDeps = qdrantLayer.pipe(
  Layer.provide(embeddingProviderLayer),
  Layer.provide(chunkingLayer),
);

// PythonApiClient always needed for sparse embedding (BM25) and chunking
const fullLayer = Layer.mergeAll(qdrantWithDeps, embeddingProviderLayer, chunkingLayer).pipe(
  Layer.provideMerge(PythonApiClient.Default),
);

Effect.runPromise(
  program.pipe(
    Effect.provide(fullLayer),
    Effect.catchAllCause((cause) => Effect.logError(cause)),
  ),
);
