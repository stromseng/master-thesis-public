import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export const CONFIG = {
  // Base URLs
  BASE_URL: "https://www.shititong.cn",
  SEARCH_URL:
    "https://www.shititong.cn/page/tiku.html?cc=%25E8%2588%25AA%25E6%25B5%25B7%25E8%258B%25B1%25E8%25AF%25AD",

  // Search keywords for maritime exams (Chinese)
  // Based on filter analysis - keeping keywords with >=50% relevance
  SEARCH_KEYWORDS: [
    // High performers (>80% relevant)
    "船舶", // Ship/Vessel - 100% relevant
    "海事", // Maritime - 98.81% relevant
    "航海英语", // Navigation English - 89.29% relevant
    "轮机英语", // Marine engineering English - 86.11% relevant
    "船艇", // Boats/vessels - 83.33% relevant
    "海船", // Seagoing vessel - 81.08% relevant
    // Medium performers (50-80% relevant)
    "水手", // Sailor - 59.57% relevant
    "轮机", // Marine Engineering - 56.32% relevant
    "海员", // Seafarer - 54.32% relevant
    "船员适任", // Crew competency certification - 50% relevant
  ],
  // Removed keywords (<50% relevant):
  //   航运 (Shipping) - 30.38% relevant
  //   远洋 (Ocean-going) - 24.14% relevant
  //   海上安全 (Maritime safety) - 23.86% relevant
  //   驾驶 (Navigation/Driving) - overlaps with car driving exams
  //   船长 (Captain) - 38.96% relevant
  //   GMDSS (Global Maritime Distress Safety System) - 37.14% relevant
  //   二副 (Second Officer) - 34.52% relevant
  //   大副 (Chief Officer) - 31.46% relevant
  //   三副 (Third Officer) - 30.38% relevant
  //   机工 (Engine crew) - 14.77% relevant
  //   值班水手 (Watchkeeping sailor) - 11.9% relevant
  //   甲板 (Deck) - 11.63% relevant
  //   值班机工 (Watchkeeping engine crew) - 1.22% relevant

  // Paths
  COOKIES_PATH: join(__dirname, "cookies.json"),
  OUTPUT_DIR: join(__dirname, "../../output"),
  EXAM_LINKS_FILE: join(__dirname, "../../output/exam-links.json"),
  FILTERED_EXAM_LINKS_FILE: join(__dirname, "../../output/filtered-exam-links.json"),
  FILTER_ANALYSIS_FILE: join(__dirname, "../../output/filter-analysis.json"),
  CHAPTER_LINKS_FILE: join(__dirname, "../../output/chapter-links.json"),
  QUESTIONS_DIR: join(__dirname, "../../output/questions"),

  // Scraping settings
  REQUEST_DELAY_MS: 1000, // Delay between requests
  PAGE_LOAD_TIMEOUT_MS: 30000,
  HEADLESS: false, // Set to true for production - its for if we want to see the browser or not
  CONCURRENT_TABS: 5, // Number of browser tabs to use concurrently

  // Selectors
  SELECTORS: {
    SEARCH_INPUT: "#search-keyword",
    SEARCH_BUTTON: "#zhixingsousuo",
    EXAM_LINK: 'a.view-btn[href*="cha-kan/tikuheji"]',
    CHAPTER_BUTTON: 'button.view-btn[aria-label="查看章节题目"]',
    COLLECT_POPUP_CONFIRM: "a.layui-layer-btn0",
    COLLECT_POPUP_SKIP: "a.layui-layer-btn1",
  },

  // Regex patterns
  PATTERNS: {
    TIKUID: /\/cha-kan\/tikuheji\/([A-F0-9]{32})\.html/i,
    TIKUID_FROM_URL: /([A-F0-9]{32})/i,
  },
} as const;

export type Cookie = {
  name: string;
  value: string;
  domain: string;
  path: string;
};

export type ExamLink = {
  title: string;
  url: string;
  tikuid: string;
  found_by_keywords: string[];
};

export type ExamLinksOutput = {
  collected_at: string;
  search_url: string;
  search_keywords: string[];
  total_links: number;
  links: ExamLink[];
};

export type FilteredExamLink = ExamLink & {
  relevant: boolean;
  reason?: string;
};

export type FilteredExamLinksOutput = {
  filtered_at: string;
  source_file: string;
  total_input: number;
  total_relevant: number;
  total_irrelevant: number;
  model_used: string;
  links: FilteredExamLink[];
};

export type ChapterLink = {
  zjid: string;
  url: string;
};

export type ExamWithChapters = {
  tikuid: string;
  exam_url: string;
  title?: string;
  author?: string; // undefined if "未知" (unknown) or not found
  found_by_keywords?: string[];
  chapters: ChapterLink[];
};

export type AuthorAnalysis = {
  total_with_author: number;
  total_without_author: number;
  with_author_percent: number;
  without_author_percent: number;
  authors: Array<{
    name: string;
    count: number;
  }>;
};

export type ChapterLinksOutput = {
  collected_at: string;
  total_exams: number;
  total_chapters: number;
  author_analysis: AuthorAnalysis;
  exams: ExamWithChapters[];
};

export type QuestionOption = {
  A?: string;
  B?: string;
  C?: string;
  D?: string;
  E?: string;
  F?: string;
};

export type QuestionSource = {
  exam_url: string;
  exam_title?: string;
  found_by_keywords?: string[];
};

export type Question = {
  id: number;
  question: string | null;
  options: QuestionOption;
  answer: string | null;
  type: string | null;
  hint: string | null;
  explanation: string | null;
  difficulty: string | null;
  category_id: string | null;
  original_id: string | null;
  source?: QuestionSource;
};

export type QuestionsOutput = {
  exported_at: string;
  tikuid: string;
  zjid: string;
  exam_link: string;
  exam_title?: string;
  found_by_keywords?: string[];
  chapter_link: string;
  total_questions: number;
  source: string;
  questions: Question[];
};
