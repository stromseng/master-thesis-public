import { Context, Effect, Layer, Option } from "effect";
import { DenseEmbedding } from "./Dense";
import { LateEmbedding } from "./Late";
import { SparseEmbedding } from "./Sparse";

type DenseEmbeddingInstance = typeof DenseEmbedding.Service;
type SparseEmbeddingInstance = typeof SparseEmbedding.Service;
type LateEmbeddingInstance = typeof LateEmbedding.Service;
type DenseEmbeddingLayer<E = never, R = never> = Layer.Layer<DenseEmbedding, E, R>;
type SparseEmbeddingLayer<E = never, R = never> = Layer.Layer<SparseEmbedding, E, R>;
type LateEmbeddingLayer<E = never, R = never> = Layer.Layer<LateEmbedding, E, R>;

export interface EmbeddingProviderService {
  readonly dense: Option.Option<typeof DenseEmbedding.Service>;
  readonly sparse: Option.Option<typeof SparseEmbedding.Service>;
  readonly late: Option.Option<typeof LateEmbedding.Service>;
}

export class EmbeddingProvider extends Context.Tag("@app/EmbeddingProvider")<
  EmbeddingProvider,
  EmbeddingProviderService
>() {
  static make<DenseE, DenseR, LateE = never, LateR = never>(
    dense: DenseEmbeddingLayer<DenseE, DenseR>,
    sparse: undefined,
    late?: LateEmbeddingLayer<LateE, LateR>,
  ): Layer.Layer<EmbeddingProvider, DenseE | LateE, DenseR | LateR>;
  static make<SparseE, SparseR, LateE = never, LateR = never>(
    dense: undefined,
    sparse: SparseEmbeddingLayer<SparseE, SparseR>,
    late?: LateEmbeddingLayer<LateE, LateR>,
  ): Layer.Layer<EmbeddingProvider, SparseE | LateE, SparseR | LateR>;
  static make<DenseE, DenseR, SparseE, SparseR, LateE = never, LateR = never>(
    dense: DenseEmbeddingLayer<DenseE, DenseR>,
    sparse: SparseEmbeddingLayer<SparseE, SparseR>,
    late?: LateEmbeddingLayer<LateE, LateR>,
  ): Layer.Layer<EmbeddingProvider, DenseE | SparseE | LateE, DenseR | SparseR | LateR>;
  static make(
    dense: DenseEmbeddingInstance,
    sparse: undefined,
    late?: LateEmbeddingInstance,
  ): Layer.Layer<EmbeddingProvider>;
  static make(
    dense: undefined,
    sparse: SparseEmbeddingInstance,
    late?: LateEmbeddingInstance,
  ): Layer.Layer<EmbeddingProvider>;
  static make(
    dense: DenseEmbeddingInstance,
    sparse: SparseEmbeddingInstance,
    late?: LateEmbeddingInstance,
  ): Layer.Layer<EmbeddingProvider>;
  static make(
    dense?: DenseEmbeddingInstance | DenseEmbeddingLayer<any, any>,
    sparse?: SparseEmbeddingInstance | SparseEmbeddingLayer<any, any>,
    late?: LateEmbeddingInstance | LateEmbeddingLayer<any, any>,
  ): Layer.Layer<EmbeddingProvider, any, any> {
    if (!dense && !sparse) {
      throw new Error("EmbeddingProvider.make requires at least one of dense or sparse");
    }

    const anyLayerInput =
      (dense !== undefined && Layer.isLayer(dense)) ||
      (sparse !== undefined && Layer.isLayer(sparse)) ||
      (late !== undefined && Layer.isLayer(late));

    if (anyLayerInput) {
      if (
        (dense !== undefined && !Layer.isLayer(dense)) ||
        (sparse !== undefined && !Layer.isLayer(sparse)) ||
        (late !== undefined && !Layer.isLayer(late))
      ) {
        throw new Error("EmbeddingProvider.make requires all arguments to be layers or services");
      }

      const build = <ROut, E, R>(dependencies: Layer.Layer<ROut, E, R>) =>
        Layer.effect(
          EmbeddingProvider,
          Effect.gen(function* () {
            const resolvedDense = yield* Effect.serviceOption(DenseEmbedding);
            const resolvedSparse = yield* Effect.serviceOption(SparseEmbedding);
            const resolvedLate = yield* Effect.serviceOption(LateEmbedding);

            return EmbeddingProvider.of({
              dense: resolvedDense,
              sparse: resolvedSparse,
              late: resolvedLate,
            });
          }),
        ).pipe(Layer.provide(dependencies));

      if (dense && sparse && late) {
        return build(Layer.mergeAll(dense, sparse, late));
      }
      if (dense && sparse) {
        return build(Layer.merge(dense, sparse));
      }
      if (dense && late) {
        return build(Layer.merge(dense, late));
      }
      if (sparse && late) {
        return build(Layer.merge(sparse, late));
      }
      if (dense) {
        return build(dense);
      }
      if (sparse) {
        return build(sparse);
      }

      throw new Error("Unreachable: validated layer-based embedding provider configuration");
    }

    // If none of the params are layers, they are direct services, we can simply set them then.
    return Layer.succeed(
      EmbeddingProvider,
      EmbeddingProvider.of({
        dense: dense ? Option.some(dense as DenseEmbeddingInstance) : Option.none(),
        sparse: sparse ? Option.some(sparse as SparseEmbeddingInstance) : Option.none(),
        late: late ? Option.some(late as LateEmbeddingInstance) : Option.none(),
      }),
    );
  }
}
