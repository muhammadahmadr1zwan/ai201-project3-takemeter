# TakeMeter — Annotation Guide

## Your next step (Milestone 3)

You have **raw unlabeled posts** in `data/raw_posts.csv`. Your job is to read each one and assign exactly one label.

### 1. Open the raw data

Open `data/raw_posts.csv` in Excel, Google Sheets, or VS Code.

### 2. Create your labeled file

Save a copy as `data/labeled_dataset.csv` with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `text` | yes | The post/comment (don't edit unless removing PII) |
| `label` | yes | One of: `analysis`, `hot_take`, `reaction` |
| `notes` | optional | Why it was hard, or correction notes |

### 3. Label using planning.md

Keep `planning.md` open. For every post:

1. Read the full text
2. Apply the one-sentence definition
3. If borderline, use the edge-case decision rules
4. Write hard cases in `notes` and in the planning.md annotation log

**Do not skim.** The model learns your labels — noisy annotation = noisy model.

### 4. Balance check

After 200+ labels, count per class:

```bash
python scripts/check_balance.py data/labeled_dataset.csv
```

Target: **no label above 70%**, ideally **≥20% each** (~40+ per label).

If imbalanced, collect more from underrepresented types:
- **analysis:** long comments in game threads, OC self-posts
- **hot_take:** trade rumors, ranking debates, "unpopular opinion" threads
- **reaction:** post-game threads, highlight submissions

### 5. Optional pre-labeling (disclose in README)

If you have a Groq API key:

```powershell
$env:GROQ_API_KEY = "your_key"
python scripts/prelabel.py
```

Review **every** pre-label in `data/prelabeled.csv`, fix mistakes, save as `labeled_dataset.csv`.

### 6. Manual collection (also valid)

The course recommends copy-paste from Reddit directly (~1–2 hours). Browse:

- https://www.reddit.com/r/nba/new/
- Post-game threads
- Trade/rumor threads

Add rows to your CSV. Public posts only.

---

## Label quick reference

| Label | Ask yourself |
|-------|--------------|
| `analysis` | Is there structured reasoning with verifiable evidence? |
| `hot_take` | Is it a bold claim that asserts without really arguing? |
| `reaction` | Is it mostly emotional venting/excitement about a moment? |

## After labeling

1. Upload `labeled_dataset.csv` to your Colab notebook
2. Run baseline (Section 5) **before** fine-tuning
3. Fine-tune (Sections 3–4)
4. Download `evaluation_results.json` and `confusion_matrix.png` to `outputs/`
