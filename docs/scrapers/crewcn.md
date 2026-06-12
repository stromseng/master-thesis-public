# CrewCN Question Extraction

Extract maritime exam questions from DOCX/PDF files sourced from crewcn.com.

## Quick Start

```bash
# From project root
cd code/python && uv run python scripts/crewcn/extract_questions.py

# Or manually
cd code/python && uv run python scripts/crewcn/extract_questions.py
```

## Step-by-Step Guide

### Step 1: Ensure Dependencies

Make sure `python-docx` and `pymupdf` are installed:

```bash
cd code/python
uv sync
```

### Step 2: Place Input Files

Put source files in `files/` directory:

| File | Expected | Format | Parser |
|------|----------|--------|--------|
| dalian_2580.md (auto-converted from DOCX via pandoc) | ~2145 | `KEY: X` after options | KeyFormatParser |
| Latest Sailing English (Chinese and English versions)...pdf | ~866 | 3-column PDF layout | PDFColumnFormatParser |
| Latest Sailing English Question Bank...3300.docx | ~136 | `KEY: X` after options | SimpleKeyFormatParser |
| Seamen's 62nd...docx | ~95 | Orange text = answer | ColorFormatParser |

### Step 3: Run Extraction

```bash
cd code/python && uv run python scripts/crewcn/extract_questions.py
```

### Step 4: Check Output

Output files are written to `output/`:

- Individual JSON files per source document
- `all-questions.json` - combined output

## Output Format

Each question follows this structure:

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
  "hint": null,
  "explanation": null,
  "source": {
    "parent_url": "https://www.crewcn.com/download/?Smallclassname=13",
    "exam_title": "Dalian Maritime University English 2580",
    "file_name": "dalian_2580.md"
  }
}
```

## Directory Structure

```
code/python/scripts/crewcn/
├── README.md              # This file
├── extract_questions.py   # Main entry point
├── parsers/
│   ├── base.py            # Base parser class
│   ├── key_format.py      # For "KEY: X" format (Dalian)
│   ├── simple_key_format.py # For 3300 format
│   ├── embedded_answer.py # For "__X__" format
│   ├── color_format.py    # For orange text format (62nd)
│   └── pdf_column_format.py # For 3-column PDF (Latest Sailing)
├── files/                 # Input DOCX/PDF files
└── output/                # Output JSON files
```

## Current Results

| Source | Expected | Valid | Extracted | Success Rate |
|--------|----------|-------|-----------|--------------|
| Latest Sailing (PDF) | ~866 | 792 | 810 | 98% |
| Dalian 2580 (md) | ~2145 | 2049 | 2093 | 98% |
| 3300 | ~136 | 135 | 135 | 100% |
| 62nd | ~95 | 82 | 93 | 88% |
| **Total** | **~3242** | **3058** | | |

**Notes:**
- *Expected* = question count implied by filename
- *Extracted* = questions parsed from document
- *Valid* = questions with complete options and answer
- 3300 only has 135 `KEY:` lines in the document (the "3300" likely refers to an exam code)
- Dalian DOCX is converted to markdown via pandoc to preserve Word's list numbering
- "同上" (same as above) duplicate entries are filtered out for a cleaner dataset

## Troubleshooting

### Missing dependencies

```bash
cd code/python && uv add python-docx pymupdf
```

### No questions extracted

- Check that input files are in `files/` directory
- Verify file names match expected patterns in `extract_questions.py`

### Low extraction rate

Some documents have formatting issues in the source:
- Missing options (A/B/C)
- No answer marking
- Malformed question structure

Check the plan document at `docs/crewcn_extraction_plan.md` for detailed analysis.
