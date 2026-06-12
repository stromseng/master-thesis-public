import { describe, expect, it } from "@effect/vitest";
import { Effect, Option } from "effect";
import {
  CLASSIFICATION_MODE_DEFINITIONS,
  MULTIMODAL_REASONING_CATEGORIES,
  SECONDARY_FACETS,
  TEXT_REASONING_CATEGORIES,
  TOPIC_CATEGORIES,
  REASONING_CATEGORY_DEFINITIONS,
  aggregateClassificationStats,
  buildClassificationAggregates,
  normalizeClassification,
  parseSamplePercent,
  sampleQuestionsByPercentage,
  selectDatasets,
  validateResumeCheckpoint,
  type QuestionClassificationResult,
} from "../../scripts/analysis/classify-question-categories";

const makeResult = (
  overrides: Partial<QuestionClassificationResult> = {},
): QuestionClassificationResult => ({
  sourceDataset: "pei2024-uk",
  groupId: "group-1",
  questionKey: "pei2024-uk::question-1",
  questionId: "question-1",
  questionText: "What rule applies in a crossing situation?",
  optionCount: 4,
  hasImages: false,
  questionMetadata: {},
  classificationMode: "text_only",
  classification: {
    primaryTopic: "collision_avoidance_and_colregs",
    secondaryFacets: ["lights_shapes_buoys_and_sound_signals"],
    reasoningType: "rule_application",
  },
  ...overrides,
});

describe("normalizeClassification", () => {
  it("deduplicates secondary facets and caps them at three items", () => {
    const normalized = normalizeClassification({
      primaryTopic: "collision_avoidance_and_colregs",
      secondaryFacets: [
        "lights_shapes_buoys_and_sound_signals",
        "lights_shapes_buoys_and_sound_signals",
        "radar_arpa_ecdis_gnss_and_instrumentation",
        "charts_publications_and_notices",
        "mooring_anchoring_and_berthing",
      ],
      reasoningType: "rule_application",
    });

    expect(normalized.secondaryFacets).toEqual([
      "lights_shapes_buoys_and_sound_signals",
      "radar_arpa_ecdis_gnss_and_instrumentation",
      "charts_publications_and_notices",
    ]);
  });
});

describe("buildClassificationAggregates", () => {
  it("builds dense per-dataset distributions and keeps dataset summaries", () => {
    const aggregates = buildClassificationAggregates([
      makeResult(),
      makeResult({
        sourceDataset: "crewcn",
        questionKey: "crewcn::question-2",
        questionId: "question-2",
        classification: {
          primaryTopic: "marine_engineering_and_propulsion",
          secondaryFacets: ["cargo_operations"],
          reasoningType: "factual_recall",
        },
      }),
    ]);

    expect(aggregates.totals.overallQuestions).toBe(2);
    expect(aggregates.totals.byDataset).toEqual([
      {
        datasetId: "crewcn",
        totalQuestions: 1,
      },
      {
        datasetId: "pei2024-uk",
        totalQuestions: 1,
      },
    ]);

    expect(aggregates.distributions.primaryTopic.overall.counts).toContainEqual({
      category: "collision_avoidance_and_colregs",
      count: 1,
      percentage: 50,
    });
    expect(aggregates.distributions.primaryTopic.overall.counts).toContainEqual({
      category: "marine_engineering_and_propulsion",
      count: 1,
      percentage: 50,
    });
    expect(aggregates.distributions.primaryTopic.overall.counts).toContainEqual({
      category: "corrupted_or_prompt_leakage",
      count: 0,
      percentage: 0,
    });

    expect(aggregates.secondaryFacetUsage.overall.totalFacetAssignments).toBe(2);
    expect(aggregates.secondaryFacetUsage.overall.counts).toContainEqual({
      category: "cargo_operations",
      count: 1,
      percentageOfQuestions: 50,
      percentageOfFacetAssignments: 50,
    });
  });
});

describe("aggregateClassificationStats", () => {
  it("builds overall and per-dataset counts", () => {
    const stats = aggregateClassificationStats([
      makeResult(),
      makeResult({
        questionId: "question-2",
        classification: {
          primaryTopic: "marine_engineering_and_propulsion",
          secondaryFacets: [],
          reasoningType: "factual_recall",
        },
        sourceDataset: "crewcn",
      }),
      makeResult({
        questionId: "question-3",
        classificationMode: "multimodal",
        classification: {
          primaryTopic: "collision_avoidance_and_colregs",
          secondaryFacets: ["radar_arpa_ecdis_gnss_and_instrumentation"],
          reasoningType: "spatial_reasoning",
        },
        sourceDataset: "navreas-scene-understanding",
      }),
    ]);

    expect(stats.overall.totalQuestions).toBe(3);
    expect(stats.overall.byClassificationMode).toEqual([
      { category: "text_only", count: 2, percentage: 66.7 },
      { category: "multimodal", count: 1, percentage: 33.3 },
    ]);
    expect(stats.overall.byPrimaryTopic).toEqual([
      { category: "collision_avoidance_and_colregs", count: 2, percentage: 66.7 },
      { category: "marine_engineering_and_propulsion", count: 1, percentage: 33.3 },
    ]);
    expect(stats.byDataset["pei2024-uk"]?.byReasoningType).toEqual([
      { category: "rule_application", count: 1, percentage: 100 },
    ]);
    expect(stats.byDataset["navreas-scene-understanding"]?.byReasoningType).toEqual([
      { category: "spatial_reasoning", count: 1, percentage: 100 },
    ]);
  });
});

describe("validateResumeCheckpoint", () => {
  it.effect("accepts matching checkpoint metadata", () =>
    validateResumeCheckpoint(
      "/tmp/checkpoint.json",
      {
        generatedAt: "2026-04-08T00:00:00.000Z",
        sourceDatasets: ["crewcn", "pei2024-uk"],
        samplePercentPerDataset: 100,
        limitPerDataset: 0,
        batchSize: 2,
        topicCategories: TOPIC_CATEGORIES,
        secondaryFacets: SECONDARY_FACETS,
        textReasoningCategories: TEXT_REASONING_CATEGORIES,
        multimodalReasoningCategories: MULTIMODAL_REASONING_CATEGORIES,
        reasoningCategories: REASONING_CATEGORY_DEFINITIONS,
        classificationModes: CLASSIFICATION_MODE_DEFINITIONS,
      },
      {
        sourceDatasets: ["pei2024-uk", "crewcn"],
        samplePercentPerDataset: 100,
        limitPerDataset: 0,
        batchSize: 2,
        topicCategoryIds: TOPIC_CATEGORIES.map((category) => category.id),
        secondaryFacetIds: SECONDARY_FACETS.map((facet) => facet.id),
        reasoningCategoryIds: REASONING_CATEGORY_DEFINITIONS.map((category) => category.id),
      },
    ),
  );
});

describe("parseSamplePercent", () => {
  it.effect("parses an omitted percentage as one hundred", () =>
    Effect.gen(function* () {
      const value = yield* parseSamplePercent(Option.none());
      expect(value).toBe(100);
    }),
  );

  it.effect("accepts decimals between 0 and 100", () =>
    Effect.gen(function* () {
      const value = yield* parseSamplePercent(Option.some("2.5"));
      expect(value).toBe(2.5);
    }),
  );

  it.effect("fails for invalid percentages", () =>
    Effect.gen(function* () {
      const result = yield* parseSamplePercent(Option.some("120")).pipe(Effect.either);
      expect(result._tag).toBe("Left");
    }),
  );
});

describe("sampleQuestionsByPercentage", () => {
  it("deterministically samples each dataset subset size", () => {
    const items = Array.from({ length: 10 }, (_, index) => ({
      sourceDataset: "pei2024-uk",
      question: { id: `q-${index + 1}` },
    }));

    const sampled = sampleQuestionsByPercentage(items, 20);
    const sampledAgain = sampleQuestionsByPercentage(items, 20);

    expect(sampled).toHaveLength(2);
    expect(sampled.map((item) => item.question.id)).toEqual(
      sampledAgain.map((item) => item.question.id),
    );
  });

  it("keeps at least one item when the percentage is positive", () => {
    const items = Array.from({ length: 3 }, (_, index) => ({
      sourceDataset: "pei2024-uk",
      question: { id: `q-${index + 1}` },
    }));

    const sampled = sampleQuestionsByPercentage(items, 1);

    expect(sampled).toHaveLength(1);
  });
});

describe("selectDatasets", () => {
  it.effect("returns all datasets when no dataset filter is provided", () =>
    Effect.gen(function* () {
      const datasets = yield* selectDatasets([]);
      expect(datasets.length).toBeGreaterThan(10);
    }),
  );

  it.effect("fails on unknown dataset IDs", () =>
    Effect.gen(function* () {
      const result = yield* selectDatasets(["not-a-dataset"]).pipe(Effect.either);
      expect(result._tag).toBe("Left");
    }),
  );
});
