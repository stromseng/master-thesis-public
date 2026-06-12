/**
 * Test script demonstrating Effect runtime introspection
 * Shows how to extract information about built layers and services
 *
 * Key findings:
 * - Runtime has: context, runtimeFlags (number), fiberRefs
 * - Context has: unsafeMap (Map<string, any>) containing all services by tag key
 * - Tag has: key (string identifier like "@app/ServiceName")
 */
import { Context, Effect, Layer, Runtime, RuntimeFlags, Schema } from "effect";

/** Schema-compliant JSON encoding */
const toJson = Schema.encode(Schema.parseJson(Schema.Unknown));

// ============================================================================
// Example Services (similar to your codebase)
// ============================================================================

class DatabaseConfig extends Context.Tag("@app/DatabaseConfig")<
  DatabaseConfig,
  { readonly url: string; readonly poolSize: number }
>() {}

class HttpConfig extends Context.Tag("@app/HttpConfig")<
  HttpConfig,
  { readonly baseUrl: string; readonly timeout: number }
>() {}

class ModelConfig extends Context.Tag("@app/ModelConfig")<
  ModelConfig,
  { readonly modelId: string; readonly temperature: number }
>() {}

// ============================================================================
// Layer Metadata Registry - Track what layers are built
// ============================================================================

interface LayerMetadata {
  name: string;
  description?: string;
  config?: Record<string, unknown>;
}

class LayerRegistry extends Context.Tag("@app/LayerRegistry")<
  LayerRegistry,
  { readonly layers: Map<string, LayerMetadata> }
>() {}

/**
 * Creates a layer that also registers metadata about itself
 */
const layerWithMetadata = <A, E, R>(
  metadata: LayerMetadata,
  layer: Layer.Layer<A, E, R>,
): Layer.Layer<A, E, R | LayerRegistry> =>
  Layer.effectDiscard(
    Effect.gen(function* () {
      const registry = yield* Effect.serviceOption(LayerRegistry);
      if (registry._tag === "Some") {
        registry.value.layers.set(metadata.name, metadata);
      }
    }),
  ).pipe(Layer.provideMerge(layer));

// ============================================================================
// Service Tag Introspection
// ============================================================================

/**
 * Get the identifier key from a Context.Tag
 */
const getTagKey = <Id, Service>(tag: Context.Tag<Id, Service>): string => {
  return tag.key as string;
};

/**
 * Check if a runtime has a specific service by tag
 */
const hasService = <R, Id, Service>(
  runtime: Runtime.Runtime<R>,
  tag: Context.Tag<Id, Service>,
): boolean => {
  return Context.getOption(runtime.context, tag as any)._tag === "Some";
};

/**
 * Get all known service tags and check which are present
 */
const introspectServices = <R>(
  runtime: Runtime.Runtime<R>,
  knownTags: Context.Tag<any, any>[],
): { key: string; present: boolean }[] => {
  return knownTags.map((tag) => ({
    key: getTagKey(tag),
    present: hasService(runtime, tag),
  }));
};

/**
 * Get ALL service keys from a runtime's context (using unsafeMap)
 */
const getAllServiceKeys = <R>(runtime: Runtime.Runtime<R>): string[] => {
  // Context.unsafeMap is a Map<string, any> of tag key -> service
  return Array.from(runtime.context.unsafeMap.keys());
};

// ============================================================================
// Runtime Information Extraction
// ============================================================================

interface RuntimeInfo {
  allServiceKeys: string[];
  knownServices: { key: string; present: boolean }[];
  runtimeFlags: number;
  runtimeFlagsDecoded: Record<string, boolean>;
}

const extractRuntimeInfo = <R>(
  runtime: Runtime.Runtime<R>,
  knownTags: Context.Tag<any, any>[],
): RuntimeInfo => {
  const flags = runtime.runtimeFlags;

  return {
    allServiceKeys: getAllServiceKeys(runtime),
    knownServices: introspectServices(runtime, knownTags),
    runtimeFlags: flags,
    runtimeFlagsDecoded: {
      // Decode runtime flags using RuntimeFlags module
      cooperativeYielding: RuntimeFlags.cooperativeYielding(flags),
      interruptible: RuntimeFlags.interruptible(flags),
      runtimeMetrics: RuntimeFlags.runtimeMetrics(flags),
      opSupervision: RuntimeFlags.opSupervision(flags),
      windDown: RuntimeFlags.windDown(flags),
    },
  };
};

// ============================================================================
// Demo Program
// ============================================================================

// Create some example layers
const dbConfigLayer = Layer.succeed(DatabaseConfig, {
  url: "postgres://localhost:5432/db",
  poolSize: 10,
});

const httpConfigLayer = Layer.succeed(HttpConfig, {
  baseUrl: "https://api.example.com",
  timeout: 5000,
});

const modelConfigLayer = Layer.succeed(ModelConfig, {
  modelId: "gpt-4",
  temperature: 0.7,
});

// Registry layer
const registryLayer = Layer.succeed(LayerRegistry, {
  layers: new Map<string, LayerMetadata>(),
});

// Combine layers with metadata tracking
const trackedDbLayer = layerWithMetadata(
  { name: "DatabaseConfig", description: "PostgreSQL connection config", config: { poolSize: 10 } },
  dbConfigLayer,
);

const trackedHttpLayer = layerWithMetadata(
  { name: "HttpConfig", description: "HTTP client configuration", config: { timeout: 5000 } },
  httpConfigLayer,
);

const trackedModelLayer = layerWithMetadata(
  { name: "ModelConfig", description: "LLM model configuration", config: { modelId: "gpt-4" } },
  modelConfigLayer,
);

// All known service tags for introspection (cast to work with introspection functions)
const knownServiceTags: Context.Tag<any, any>[] = [
  DatabaseConfig as Context.Tag<any, any>,
  HttpConfig as Context.Tag<any, any>,
  ModelConfig as Context.Tag<any, any>,
  LayerRegistry as Context.Tag<any, any>,
];

// Full layer stack
const fullLayer = Layer.mergeAll(trackedDbLayer, trackedHttpLayer, trackedModelLayer).pipe(
  Layer.provideMerge(registryLayer),
);

// Partial layer stack (missing ModelConfig)
const partialLayer = Layer.mergeAll(trackedDbLayer, trackedHttpLayer).pipe(
  Layer.provideMerge(registryLayer),
);

const program = Effect.gen(function* () {
  console.log("=== Effect Runtime Introspection Demo ===\n");

  // Get runtime for full layer
  const fullRuntime = yield* Effect.runtime<Layer.Layer.Success<typeof fullLayer>>();

  console.log("1. Service Tag Keys (predefined):");
  for (const tag of knownServiceTags) {
    console.log(`   - ${getTagKey(tag)}`);
  }

  console.log("\n2. Runtime Info (Full Layer):");
  const fullInfo = extractRuntimeInfo(fullRuntime, knownServiceTags);

  console.log("   All services in context (from unsafeMap):");
  for (const key of fullInfo.allServiceKeys) {
    console.log(`   - ${key}`);
  }

  console.log("\n   Known services status:");
  for (const svc of fullInfo.knownServices) {
    console.log(`   - ${svc.key}: ${svc.present ? "✓" : "✗"}`);
  }

  console.log("\n   Runtime Flags:", fullInfo.runtimeFlags);
  console.log("   Runtime Flags Decoded:", fullInfo.runtimeFlagsDecoded);

  // Access registry to see tracked layers
  const registry = yield* LayerRegistry;
  console.log("\n3. Registered Layer Metadata:");
  for (const [name, meta] of registry.layers) {
    console.log(`   - ${name}: ${meta.description}`);
    if (meta.config) {
      console.log(`     Config: ${toJson(meta.config)}`);
    }
  }

  // Access actual service values
  console.log("\n4. Service Values:");
  const dbConfig = yield* DatabaseConfig;
  const httpConfig = yield* HttpConfig;
  const modelConfig = yield* ModelConfig;
  console.log(`   DatabaseConfig: ${toJson(dbConfig)}`);
  console.log(`   HttpConfig: ${toJson(httpConfig)}`);
  console.log(`   ModelConfig: ${toJson(modelConfig)}`);

  // Build metadata object for experiment
  const experimentMetadata = {
    allServices: fullInfo.allServiceKeys,
    knownServices: Object.fromEntries(fullInfo.knownServices.map((s) => [s.key, s.present])),
    layerConfig: Object.fromEntries(registry.layers.entries()),
    runtimeFlags: fullInfo.runtimeFlagsDecoded,
    modelConfig: {
      modelId: modelConfig.modelId,
      temperature: modelConfig.temperature,
    },
  };

  console.log("\n5. Metadata for Experiment Run:");
  yield* Effect.log(experimentMetadata);

  return experimentMetadata;
});

// Also demonstrate partial layer introspection
const partialProgram = Effect.gen(function* () {
  console.log("\n=== Partial Layer Demo (Missing ModelConfig) ===\n");

  const partialRuntime = yield* Effect.runtime<Layer.Layer.Success<typeof partialLayer>>();

  console.log("All services in partial context:");
  const partialInfo = extractRuntimeInfo(partialRuntime, knownServiceTags);
  for (const key of partialInfo.allServiceKeys) {
    console.log(`   - ${key}`);
  }

  console.log("\nKnown service availability:");
  for (const svc of partialInfo.knownServices) {
    console.log(`   - ${svc.key}: ${svc.present ? "✓" : "✗"}`);
  }

  // Optional service access pattern
  const modelOpt = yield* Effect.serviceOption(ModelConfig);
  console.log(`\n   ModelConfig via serviceOption: ${modelOpt._tag}`);
});

// Run both demos
const main = Effect.gen(function* () {
  yield* program.pipe(Effect.provide(fullLayer));
  yield* partialProgram.pipe(Effect.provide(partialLayer));
});

Effect.runPromise(main).catch(console.error);
