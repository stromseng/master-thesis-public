# Shititong.cn Exam Question Scraper

Scrapes maritime exam questions from shititong.cn using Playwright.

## Prerequisites

1. **Node.js/Bun** - This project uses Bun as the runtime
2. **Playwright** - Browser automation library
3. **Authentication cookies** - Required for accessing the exam content

## Setup

### 1. Install dependencies

```bash
cd code/ts
bun install
```

### 2. Install Playwright browsers

```bash
bunx playwright install chromium
```

### 3. Set up authentication cookies

The scraper requires authenticated session cookies to access exam content. You need a valid account on shititong.cn.

1. Log in to https://www.shititong.cn with your account
2. Extract your session cookies using browser DevTools
3. Create `code/ts/scripts/shititong/cookies.json` with your cookie values in Playwright cookie format (see `cookies.example.json` as a template)

## Usage

Run each phase sequentially. All scripts support `--debug` for verbose logging (by default they show a progress bar).

### Phase 1: Collect Exam Links

Searches for maritime exam packages and saves all exam links with titles and keyword tracking.

```bash
bun run scrape:1-exams           # Progress bar mode
bun run scrape:1-exams --debug   # Verbose logging
```

**Output:** `code/ts/output/exam-links.json`

### Phase 1.5: Filter by Relevance (Optional)

Uses LLM to filter out irrelevant exam datasets based on titles. This step is recommended when the search returns many unrelated results (e.g., aviation exams, driving license exams).

Note: The systemprompt is saved in `code/ts/scripts/shititong/filter-exams.ts` - line 217 function `classifyBatch()`, you can customize it as needed.

**Requires:** `LITE_LLM_API_KEY` environment variable

```bash
bun run scrape:2-filter            # Progress bar mode
bun run scrape:2-filter --debug    # Verbose logging
```

**Output:**
- `code/ts/output/filtered-exam-links.json` - Filtered links with relevance flags
- `code/ts/output/filter-analysis.json` - Analysis of keyword effectiveness

The filter classifies each dataset as relevant or irrelevant for maritime English exams. Irrelevant datasets include:
- Aviation/aircraft exams (飞机, 航空, 南航)
- Driver's license exams (驾驶证, 驾照, 科目)
- Medical, construction, or IT exams
- General English exams not specific to maritime

### Phase 2: Collect Chapter Links

For each exam link, extracts all chapter/section links and the author (题库作者) if available. Authors marked as "未知" (unknown) are treated as no author.

```bash
bun run scrape:3-chapters            # Uses filtered file if available
bun run scrape:3-chapters --all      # Force use unfiltered exam-links.json
bun run scrape:3-chapters --filtered # Force use filtered-exam-links.json
bun run scrape:3-chapters --debug    # Verbose logging (includes detailed author analysis)
```

**Output:** `code/ts/output/chapter-links.json` - includes author analysis with statistics on how many exam packages have known authors vs unknown.

### Phase 3: Extract Questions

Navigates to each chapter and extracts all questions.

```bash
bun run scrape:4-questions           # Progress bar mode
bun run scrape:4-questions --debug   # Verbose logging
```

**Output:** `code/ts/output/questions/*.json` (one file per chapter)

### Phase 3.5: Fix True/False Questions

The website doesn't store options for 判断题 (True/False) questions. This script adds default options to those questions so they aren't filtered out during merge.

```bash
bun run scrape:fix-tf               # Progress bar mode
bun run scrape:fix-tf --debug       # Verbose logging
```

This adds the following options to True/False questions with empty options:
- A: 正确 (Correct/True)
- B: 错误 (Wrong/False)

**Modifies:** `code/ts/output/questions/*.json` (in place)

### Phase 4: Merge Questions

Merges all question files, removes duplicates, and filters out questions with empty options.

```bash
bun run scrape:5-merge               # Progress bar mode
bun run scrape:5-merge --debug       # Verbose logging
```

**Output:** `code/ts/output/all-questions.json`

### Phase 5: Split by Language (Optional)

Splits questions into three files based on character detection:
- **Chinese**: Question text contains Chinese characters
- **Mixed**: English question text but Chinese options
- **English**: No Chinese characters anywhere

```bash
bun run scrape:6-split               # Progress bar mode
bun run scrape:6-split --debug       # Verbose logging
```

**Output:**
- `code/ts/output/chinese-questions.json` - Questions with Chinese question text
- `code/ts/output/mixed-questions.json` - English questions with Chinese options
- `code/ts/output/english-questions.json` - Pure English questions

## Configuration

Edit `code/ts/scripts/shititong/config.ts` to customize:

- `SEARCH_KEYWORDS` - Chinese keywords to search for exam packages
- `REQUEST_DELAY_MS` - Delay between requests (default: 1000ms)
- `HEADLESS` - Set to `true` for headless mode, `false` to see the browser
- `CONCURRENT_TABS` - Number of browser tabs for chapter collection (default: 3)

## Output Format

### exam-links.json
```json
{
  "collected_at": "2024-01-21T10:00:00Z",
  "search_url": "https://www.shititong.cn/page/tiku.html?cc=...",
  "search_keywords": ["航海英语", "海事", ...],
  "total_links": 519,
  "links": [
    {
      "title": "航海英语考试题目",
      "url": "https://www.shititong.cn/cha-kan/tikuheji/CB44A99ED24000015AA6A2001BF01FA6.html",
      "tikuid": "CB44A99ED24000015AA6A2001BF01FA6",
      "found_by_keywords": ["航海英语", "海事", "海员"]
    }
  ]
}
```

### filtered-exam-links.json
```json
{
  "filtered_at": "2024-01-21T10:30:00Z",
  "source_file": "code/ts/output/exam-links.json",
  "total_input": 519,
  "total_relevant": 245,
  "total_irrelevant": 274,
  "model_used": "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",
  "links": [
    {
      "title": "航海英语考试题目",
      "url": "https://www.shititong.cn/cha-kan/tikuheji/...",
      "tikuid": "CB44A99ED24000015AA6A2001BF01FA6",
      "found_by_keywords": ["航海英语", "海事"],
      "relevant": true,
      "reason": "Maritime English exam - directly relevant"
    },
    {
      "title": "南航机型英语",
      "url": "https://www.shititong.cn/cha-kan/tikuheji/...",
      "tikuid": "CA6982BDEFE0000154681CEDC68C12E0",
      "found_by_keywords": ["航海英语"],
      "relevant": false,
      "reason": "Aviation exam for South Airlines - not maritime"
    }
  ]
}
```

### filter-analysis.json
```json
{
  "analyzed_at": "2024-01-21T10:30:00Z",
  "source_file": "code/ts/output/filtered-exam-links.json",
  "summary": {
    "total_links": 519,
    "total_relevant": 245,
    "total_irrelevant": 274,
    "relevant_percent": 47.21,
    "irrelevant_percent": 52.79
  },
  "keyword_analysis": [
    {
      "keyword": "航海英语",
      "total_links": 150,
      "relevant": 140,
      "irrelevant": 10,
      "relevant_percent": 93.33,
      "irrelevant_percent": 6.67
    }
  ],
  "overlap_analysis": {
    "links_with_single_keyword": 300,
    "links_with_multiple_keywords": 219,
    "average_keywords_per_link": 1.85
  },
  "top_irrelevant_patterns": [
    {
      "pattern": "航空/飞机/南航 (Aviation)",
      "count": 45,
      "example_titles": ["南航机型英语", "航空英语考试"]
    }
  ]
}
```

### chapter-links.json
```json
{
  "collected_at": "2024-01-21T11:00:00Z",
  "total_exams": 245,
  "total_chapters": 1250,
  "author_analysis": {
    "total_with_author": 120,
    "total_without_author": 125,
    "with_author_percent": 48.98,
    "without_author_percent": 51.02,
    "authors": [
      { "name": "张三", "count": 15 },
      { "name": "李四", "count": 10 }
    ]
  },
  "exams": [
    {
      "tikuid": "CB44A99ED24000015AA6A2001BF01FA6",
      "exam_url": "https://www.shititong.cn/cha-kan/tikuheji/...",
      "title": "航海英语考试题目",
      "author": "张三",
      "found_by_keywords": ["航海英语", "海事"],
      "chapters": [
        { "zjid": "ABC123", "url": "https://..." }
      ]
    }
  ]
}
```

### Question format
```json
{
  "id": 1,
  "question": "What is the meaning of...",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "answer": "A",
  "type": "单选题",
  "hint": null,
  "explanation": null,
  "difficulty": null,
  "category_id": null,
  "original_id": "ABC123",
  "source": {
    "exam_url": "https://www.shititong.cn/cha-kan/tikuheji/CB44A99ED24000015AA6A2001BF01FA6.html",
    "exam_title": "航海英语考试题目",
    "found_by_keywords": ["航海英语", "海事"]
  }
}
```

### chinese-questions.json / mixed-questions.json / english-questions.json
```json
{
  "exported_at": "2024-01-21T12:00:00Z",
  "source": "shititong.cn",
  "language": "chinese",  // or "mixed" or "english"
  "statistics": {
    "total_questions": 1500,
    "by_type": {
      "单选题": 1200,
      "判断题": 300
    }
  },
  "questions": [...]
}
```

## Troubleshooting

### Cookies expired
Session cookies expire after some time. If you get authentication errors, log in again and update `cookies.json`.

### Rate limiting
If you get blocked, increase `REQUEST_DELAY_MS` in config.ts.

### Browser not launching
Make sure Playwright browsers are installed:
```bash
bunx playwright install chromium
```

## Notes

- The scraper runs in non-headless mode by default so you can see what's happening
- Progress is saved incrementally, so you can resume if interrupted
- Duplicate questions are automatically filtered out
- Each question in `all-questions.json` includes source metadata (`source.exam_url`, `source.exam_title`, `source.found_by_keywords`) so you can trace questions back to their original exam and see which search keywords matched
