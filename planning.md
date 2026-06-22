# TakeMeter — Planning Document

> Written before data collection. Updated as annotation and evaluation proceed.

---

## 1. Community

**Choice:** [r/nba](https://www.reddit.com/r/nba/) — the primary Reddit community for NBA discussion.

**Why this community:** r/nba is text-heavy, high-volume, and actively debates discourse quality. Regulars distinguish "real analysis" from "hot takes" and from pure emotional reactions after games. Posts vary from stat-backed breakdowns to bold unsupported claims to one-line venting — exactly the kind of variation a classifier needs.

**Why it's a good fit for classification:** The community has recognizable discourse norms. People upvote analysis threads, mock lazy hot takes, and treat post-game reaction threads differently from offseason debate. The labels map to distinctions members already make, not abstract "quality" judgments.

---

## 2. Labels

Three mutually exclusive labels grounded in how r/nba members talk about posts:

### `analysis`

**Definition:** The post makes a structured argument using specific, verifiable evidence (stats, film observations, historical comparisons, or tactical reasoning) where the evidence would still support the claim if you removed the opinion framing.

**Clear examples:**
1. "Denver's drop coverage against PnR ball handlers leaves the corner three open — they rank 28th in opponent corner 3PA rate and 26th in corner 3PT% allowed. Teams with elite pull-up shooters (Luka, Dame) exploit this repeatedly."
2. "Since the 2014 rule changes on hand-checking, league-wide ORtg has risen from 105.6 to 115.2. The 'defense is dead' narrative ignores that pace and 3PA volume also changed; per-possession efficiency gains are real but smaller than raw PPG suggests."

**Borderline example:** "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."
- Could be `analysis` ( cites a stat) or `hot_take` (accusatory framing, cherry-picked stat).
- **Decision:** → `hot_take` (see edge case rule below).

---

### `hot_take`

**Definition:** The post states a bold, confident opinion without building a real argument — it asserts rather than reasons, even if it includes a stat or name-drop that sounds credible but isn't used as part of structured reasoning.

**Clear examples:**
1. "Trade Luka now before his attitude destroys the franchise. Mark my words."
2. "The Celtics are frauds. They haven't beaten a healthy team all playoffs."

**Borderline example:** "Jokic is the best player in the league and it's not close."
- Could be `hot_take` (bold assertion) or `reaction` (if posted immediately after a 40-point game).
- **Decision:** Without game context → `hot_take`. If posted within minutes of a specific performance with no argument → `reaction`.

---

### `reaction`

**Definition:** An immediate emotional response to a specific event (a play, a game result, a trade, an injury) with little to no argument — the post is expressing a feeling in the moment rather than making a claim to be evaluated.

**Clear examples:**
1. "WHAT A BLOCK. I literally jumped off my couch."
2. "We blew a 20-point lead again. I can't do this anymore."

**Borderline example:** "Refs rigged that game, absolute joke."
- Could be `reaction` (venting) or `hot_take` (accusatory claim).
- **Decision:** → `reaction` if the post is primarily venting with no supporting evidence or reasoning. → `hot_take` if it makes a substantive accusation the author expects others to debate.

---

## 3. Hard Edge Cases

### Primary ambiguous boundary: `analysis` ↔ `hot_take`

**Type of post:** Single-stat or single-fact posts with accusatory or bold framing — enough evidence to sound credible, but not enough to constitute an argument.

**Decision rule:** If removing the opinion framing would leave evidence that genuinely supports the conclusion as part of a reasoned case, label `analysis`. If the evidence is decorative, cherry-picked, or a lone fact supporting a pre-decided take, label `hot_take`.

**Example:** "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500." → `hot_take` (one stat selected for effect; no comparison, no context, no structured reasoning).

### Secondary ambiguous boundary: `hot_take` ↔ `reaction`

**Decision rule:** If the post makes a debatable claim others would argue about → `hot_take`. If the post primarily vents emotion about a moment and isn't asking for debate → `reaction`.

### Annotation log (Milestone 3)

| # | Post snippet | Could be | Decided | Notes |
|---|--------------|----------|---------|-------|
| 1 | "LeBron is overrated — playoff win rate vs top seeds below .500" (pattern) | analysis / hot_take | hot_take | Single cherry-picked stat; no structured argument |
| 2 | "Is Aaron Gordon right, should there be more rest between playoff games?" + cites Lakers minutes | analysis / hot_take | hot_take | Raises a policy question with anecdotal evidence, not structured analysis |
| 3 | "Jokic in G7: 20/9/7 on 5-9 FG..." (stat line only) | analysis / reaction | analysis | Raw stat reporting — borderline; labeled analysis because it's factual box-score content, not emotional venting |

---

## 4. Data Collection Plan

**Source:** Public posts and top-level comments from r/nba via Reddit's public JSON API (`/r/nba/new.json`, `/r/nba/hot.json`, and comment threads).

**Target:** 220 raw examples collected → label 200+ for training.

**Per-label targets (before split):**

| Label | Target count | Rationale |
|-------|-------------|-----------|
| `analysis` | ~70 | Longer posts; collect from game threads, OC posts, comment deep-dives |
| `hot_take` | ~70 | Common in hot/new; trade and ranking threads |
| `reaction` | ~70 | Post-game threads, highlight posts, immediate reactions |

**Minimum per label:** 20% each (≥40 examples). No label should exceed 70% of the final dataset.

**If imbalanced after 200 examples:** Continue collecting from underrepresented categories — e.g., search game threads for `reaction`, filter long comments for `analysis`, browse controversial/new for `hot_take`.

**CSV format:** `text`, `label`, `notes` (optional). Single file; Colab notebook handles 70/15/15 split.

**Collection tool:** `scripts/collect_reddit.py` → outputs `data/raw_posts.csv` (unlabeled). After manual review, save as `data/labeled_dataset.csv`.

---

## 5. Evaluation Metrics

**Primary metrics:**

| Metric | Why |
|--------|-----|
| **Overall accuracy** | Easy to interpret baseline vs fine-tuned comparison on the same test set |
| **Per-class F1** | Labels are subjective and may be imbalanced; F1 balances precision and recall per class |
| **Confusion matrix** | Shows directional errors (e.g., `analysis` → `hot_take`) which map directly to label boundary problems |

**Why accuracy alone isn't enough:** A model predicting `hot_take` for everything could score well if that label is overrepresented. Per-class F1 exposes that. On a 3-class task, random guessing ≈ 33%; we need to beat both that and the Groq zero-shot baseline.

**Baseline comparison:** Groq `llama-3.3-70b-versatile` zero-shot on the locked test set — same examples, same labels, no fine-tuning. This tells us whether task-specific training actually helps.

**Optional stretch:** Confidence calibration — bucket predictions by confidence and check if higher confidence correlates with higher accuracy.

---

## 6. Definition of Success

**Useful classifier thresholds:**

| Criterion | Target |
|-----------|--------|
| Fine-tuned overall accuracy | ≥ 60% on test set (subjective 3-class task with 200 examples) |
| Beat zero-shot baseline | Fine-tuned accuracy ≥ baseline + 10 percentage points |
| No collapsed class | Each class F1 ≥ 0.45 |
| Actionable confusion | If one label pair dominates errors, we can name it and propose a fix |

**"Good enough" for a real community tool:** ≥ 65% accuracy, all class F1 ≥ 0.55, and error patterns are explainable (not random). Would deploy as a *assistive* label suggestion tool, not automated moderation — human review required for borderline cases.

**Honest failure is OK:** If fine-tuning barely beats baseline, that's a valid finding — it means labels may be too noisy or the task needs more data.

---

## 7. AI Tool Plan

### Label stress-testing (before annotation)

- **Action:** Give an LLM the three label definitions and ask it to generate 10 posts sitting on the `analysis`/`hot_take` boundary.
- **Done when:** Definitions were tightened using the decision rules above (single-stat posts → `hot_take` unless structured reasoning present).
- **If AI generates unclassifiable posts:** Revise definitions before annotating 200 examples.

### Annotation assistance

- **Decision:** Use LLM pre-labeling for a first pass, then manually review every example.
- **Tool:** Cursor / Claude with label definitions from this document.
- **Tracking:** Add column `prelabeled` (true/false) in CSV during pre-label pass; disclose in README AI usage section.
- **Rule:** Never accept a pre-label without reading the full post. Correct disagreements; note hard cases in `notes`.

### Failure analysis (after fine-tuning)

- **Action:** Export misclassified test examples; ask AI to identify patterns (sarcasm, short posts, stat-with-opinion, etc.).
- **Verification:** Manually re-read each flagged pattern — AI suggestions are hypotheses, not conclusions.
- **Output:** Include confirmed patterns in README evaluation report.

---

## 8. Training Plan (for README reference)

| Setting | Value | Notes |
|---------|-------|-------|
| Base model | `distilbert-base-uncased` | Fast, fits Colab T4, sufficient for 200 examples |
| Epochs | 3 (default) | May reduce to 2 if validation loss rises early |
| Learning rate | 2e-5 (default) | Standard for BERT fine-tuning |
| Batch size | 16 (default) | Limited by GPU memory |
| Split | 70/15/15 | Handled by starter Colab notebook |

**Hyperparameter note to document:** If validation accuracy plateaus by epoch 2, train for 2 epochs instead of 3 to reduce overfitting on small data.

---

## Changelog

- **Milestone 1–2:** Initial community, labels, and plan written.
- **Milestone 3:** *(update with label distribution and 3 hard examples)*
- **Stretch features:** *(update before starting any stretch work)*
