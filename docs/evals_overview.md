# Evals Overview

## Summary Table

| Dataset | Questions (evaluated) | Language | Modality | Key Characteristics |
|---------|----------------------|----------|----------|---------------------|
| shititong | 63,347 (8,519 EN text + 1,026 EN vision + 53,545 ZH text + 257 ZH vision) | EN, ZH | Text, Vision | Mixed maritime (engine, navigation, safety) |
| us_coast_guard | 10,414 (9,504 text + 910 multimodal) | EN | Text, Multimodal | Calculation-heavy, chart/image-based, scenario-based |
| crewcn | 3,061 | ZH | Text | Chinese crew exams, 83% fill-in-the-blank |
| pei2024 | 1,516 (814 EN + 702 ZH) | EN, ZH | Text | Regulation-focused, officer certification |
| raynor | 1,258 (974 text + 284 multimodal) | EN | Text, Multimodal | COLREGs only, 22.6% diagram-based |
| navreas | 92 | EN | Multimodal | Spatial reasoning, all image-based scenarios |
| **Total** | **79,688** | | | |

> **Note on Shititong counts:** The raw Shititong corpus contains ~278k questions (257k ZH + 21k EN). After OCR repair, deduplication, and filtering of malformed entries, the evaluated subset is 63,347. See [mcq-dedup-context.md](mcq-dedup-context.md) for the deduplication pipeline.

---

## shititong (63,347 questions)

### Evaluated Subsets

| Subset | Questions | Notes |
|--------|-----------|-------|
| shititong-en-text | 8,519 | English text-only, deduplicated |
| shititong-en-vision | 1,026 | English with images |
| shititong-zh-text | 53,545 | Chinese text-only, deduplicated |
| shititong-zh-vision | 257 | Chinese with images |

### Raw Corpus (pre-dedup)

| File | Raw questions | Notes |
|------|--------------|-------|
| SHITITONG_CHINESE_QUESTIONS | 257,396 | Chinese, OCR corruption in source |
| SHITITONG_ENGLISH_QUESTIONS | 21,144 | English |

### Question Categories (English)

| Category | % |
|----------|---|
| Engine/Machinery | 18.3% |
| Safety | 8.6% |
| Environmental/Weather | 7.6% |
| Navigation | 5.9% |
| Personnel/Organization | 5.4% |
| Cargo/Operations | 4.7% |
| Ship Equipment (radar, GPS, ECDIS) | 2.8% |
| Maritime Law/Regulations | 2.5% |
| Other | ~44% |

### Notable Characteristics

- Multiple question types: single choice, multiple choice, true/false, fill-in-the-blank
- Chinese corpus had significant OCR quality issues requiring repair and filtering
- Math content is minimal — "calculate/compute" appears mostly as vocabulary, not as tasks

---

## us_coast_guard (10,414 questions)

### Subsets

| Subset | Questions |
|--------|-----------|
| us-coast-guard-text-only | 9,504 |
| us-coast-guard-multimodal | 910 |

### Question Categories

| Category | Files | Description |
|----------|-------|-------------|
| Deck Safety | 15 | Fire, emergencies, survival equipment, stability |
| Deck General | 14 | Ship operations, regulations, equipment |
| Nav Problems - Oceans | 12 | Celestial navigation calculations |
| Nav Problems - Near Coastal | 9 | Chart-based practical navigation |
| Nav General - Near Coastal | 8 | Navigation theory, rules, weather |
| Nav Problems - Chart 12221TR | 7 | Chart-specific calculations |
| Nav General - Oceans | 7 | Celestial/ocean navigation theory |
| Nav Problems - Chart 12354TR | 6 | Chart-specific scenarios |
| Nav & Deck General-Safety | 4 | Combined topics |
| Great Lakes Topics | 4 | Regional navigation specifics |
| Deck Safety-Stability | 3 | Stability calculations |
| Rules of Road | 2 | Inland waterway regulations |
| Other | ~16 | Various minor topics |

### Question Type Breakdown

- **Calculation-heavy (25-30%)**: Celestial navigation, propeller math, fuel consumption, tidal predictions, course/bearing problems
- **Scenario-based (30-35%)**: Emergency procedures, vessel positioning, maneuvering situations
- **Pure recall (20-25%)**: Definitions, regulations, equipment specs, color codes
- **Image/chart-based (15-20%)**: NOAA chart references, buoy topmarks, vessel diagrams
- **Regulations (10-15%)**: Rules of the Road, SOLAS/MARPOL, crew requirements

### Math Content (Heavy)

- Celestial navigation (latitude from sextant observations, chronometer corrections)
- Propeller mathematics (slip, RPM, pitch)
- Tidal current predictions (set and drift)
- Distance/speed/time problems
- Cargo volume with temperature correction (API gravity, VCF)
- Stability calculations (metacentric height, trim)

---

## crewcn (3,061 questions)

### Notable Characteristics

- 83% fill-in-the-blank format
- Chinese crew certification exams
- Math content is minimal — mostly regulatory recall

---

## pei2024 (1,516 questions)

### Files

| File | Questions |
|------|-----------|
| uk_theory_test.json | 814 |
| zh_theory_test.json | 702 |

### Question Categories (UK file)

| Category | % |
|----------|---|
| Anchoring, Mooring & Berthing | 17% |
| Navigation & Seamanship | 12% |
| Collision Avoidance & Signals | 10% |
| Vessel Positioning & Direction | 10% |
| Collision Regulations & Definitions | 9% |
| Weather & Seamanship | 9% |
| Crew Duties & Responsibility | 6% |
| Cargo, Damage & Freight Law | 5% |
| Maritime Law & Contracts | 5% |
| Safety & Emergency Procedures | 3% |
| Life-saving & Rescue | 2% |
| Vessel Mechanics & Machinery | 2% |
| Other | 11% |

### Notable Characteristics

- 85% fill-in-the-blank format (UK file)
- Pure text-based — no images or diagrams
- Chinese file is heavily skewed toward collision avoidance (34%) and navigation operations (44%)
- Math content is minimal — mostly regulatory recall

---

## raynor (1,258 questions)

### Subsets

| Subset | Questions |
|--------|-----------|
| raynor-text | 974 |
| raynor-multimodal | 284 |

### Question Categories

| Category | Count | % | Diagrams |
|----------|-------|---|----------|
| Whistle/Horn Signals | 403 | 32.0% | 17.1% |
| Navigation Lights | 338 | 26.9% | 39.3% |
| General Knowledge | 180 | 14.3% | 10.6% |
| Rule Definitions & Applications | 79 | 6.3% | 2.5% |
| Navigation Situations (Meeting/Crossing/Overtaking) | 71 | 5.6% | 35.2% |
| Narrow Channel/Special Waters | 58 | 4.6% | 12.1% |
| Towing & Pushed Vessels | 50 | 4.0% | 12.0% |
| Day-Shapes | 37 | 2.9% | 51.4% |
| Fishing Vessels | 21 | 1.7% | 19.0% |
| Fog/Visibility Signals | 15 | 1.2% | 0% |
| Vessel Status (Anchored/Aground/NUC) | 6 | 0.5% | 0% |

### Notable Characteristics

- 100% multiple choice (4 options each)
- 22.6% include navigation diagrams (284 questions)
- 62% fill-in-the-blank style with "\_\_\_\_\_\_\_\_\_\_" blanks
- Exam split: Both International & Inland (72%), Inland Only (17%), International Only (11%)
- No math — pure COLREGs knowledge and application
- Question types: pure recall (23%), practical application (36%), scenario-based (41%)

---

## navreas (92 questions)

### Files

| File | Questions | Description |
|------|-----------|-------------|
| scene_understanding.eval.json | 40 | Collision risk, encounter types, stand-on determination |
| spatial_relationship_and_estimation_of_motion.eval.json | 40 | Position and motion estimation |
| colreg_compliance_and_good_seamanship.eval.json | 12 | Multi-ship solution evaluation |

### Question Categories

**Scene Understanding (40 questions)**

- Collision risk assessment (40%) — binary yes/no
- COLREGs encounter type classification (32.5%) — head-on, crossing, overtaking
- Stand-on ship determination (27.5%) — which vessel has right-of-way

**Spatial Relationship & Motion (40 questions)**

- Starboard/portside position (27.5%)
- Ahead/astern position (25%)
- Crossing relationship (17.5%)
- Approaching/receding motion (12.5%)
- Collision risk (15%)

**COLREGs Compliance (12 questions)**

- Violation analysis of proposed solutions (67%) — does a maneuver violate COLREGs?
- Solution safety selection (8%) — pick safest option from 10 proposals
- Traffic situation effect assessment (25%)

### Notable Characteristics

- All 92 questions include associated traffic visualization images
- Scenario-based with precise navigation parameters (bearings, distances, speeds, courses)
- Requires spatial/geometric reasoning — not arithmetic
- Multiple ship types: Passenger/Ro-Ro, General Cargo, Tanker
- Response formats: binary yes/no, ternary (port/starboard/neither), and complex multi-option
