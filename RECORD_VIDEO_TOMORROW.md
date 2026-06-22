# Record TakeMeter Demo Video — Pick Up Here

Everything you need is in this repo. You do **not** need this Cursor chat open.

**GitHub:** https://github.com/muhammadahmadr1zwan/ai201-project3-takemeter  
**Local folder:** `C:\Users\Ahmad\ai201-project3-takemeter`

---

## Quick start (tomorrow)

```powershell
cd C:\Users\Ahmad\ai201-project3-takemeter
pip install -r requirements.txt
python scripts/demo.py "Trade Luka now before his attitude destroys the franchise."
```

If that prints a label + confidence, you're ready to record.

---

## What this project is

**TakeMeter** — classifies r/nba posts into 3 labels:

| Label | Meaning |
|-------|---------|
| `analysis` | Stats/evidence-backed argument |
| `hot_take` | Bold opinion, asserts don't argue |
| `reaction` | Emotional in-the-moment response |

---

## Results to mention in video

| Model | Test accuracy |
|-------|---------------|
| Groq baseline (llama-3.3-70b) | **42.4%** |
| Fine-tuned DistilBERT | **51.5%** |
| Improvement | **+9.1 pp** |

**Main finding:** Model never predicts `reaction` (F1=0) — collapses to `hot_take`.

Open **README.md** during recording and scroll to:
- Evaluation Report → accuracy table
- Confusion matrix
- Wrong predictions (3 analyzed)
- Reflection

---

## Demo commands (run in terminal while recording)

```powershell
cd C:\Users\Ahmad\ai201-project3-takemeter

# 1. Likely CORRECT — analysis
python scripts/demo.py "Denver ranks 28th in opponent corner 3PA rate. Teams with elite pull-up shooters exploit drop coverage."

# 2. Likely CORRECT — hot_take
python scripts/demo.py "Trade Luka now before his attitude destroys the franchise. Mark my words."

# 3. Likely CORRECT — reaction (model may get wrong — good for demo!)
python scripts/demo.py "WHAT A BLOCK. I literally jumped off my couch."

# 4. Likely WRONG — stat one-liner (true=hot_take, model often says analysis)
python scripts/demo.py "SGA and Jdub finish the series with more assists than Jokic. SGA: 46 assists, Jokic: 41."

# 5. Extra
python scripts/demo.py "For the first time in history, there are 7 different champions in 7 NBA seasons."
```

**Narrate one correct:** "This is hot_take because it asserts without building an argument."

**Narrate one wrong:** "True label is hot_take but model saw stats and said analysis — matches our confusion matrix."

---

## Video checklist (3–5 min)

- [ ] Intro: r/nba + 3 labels
- [ ] Run 3–5 posts in terminal (label + confidence visible)
- [ ] Explain one **correct** prediction
- [ ] Explain one **incorrect** prediction
- [ ] Walk through README evaluation (accuracy, confusion matrix, reflection)

Full script: see `DEMO_SCRIPT.md`

---

## If demo fails tomorrow

Model weights are local (not on GitHub). They should be here:

```
takemeter-model/checkpoint-20/model.safetensors
```

If missing, re-run notebook once:

```powershell
cd C:\Users\Ahmad\ai201-project3-takemeter
$env:GROQ_API_KEY = (Get-Content "C:\Users\Ahmad\Downloads\codepth_groq_api_key.txt" -Raw).Trim()
python -m jupyter nbconvert --execute takemeter.ipynb --ExecutePreprocessor.timeout=3600
```

(Takes ~4 min on CPU. Only Section 3–4 needed if baseline already done.)

---

## Submit after recording

1. **GitHub link:** https://github.com/muhammadahmadr1zwan/ai201-project3-takemeter
2. **Demo video** → course portal
3. Files already in repo: `planning.md`, `labeled_dataset.csv`, `README.md`, `evaluation_results.json`, `confusion_matrix.png`

---

## Key files map

| File | Purpose |
|------|---------|
| `README.md` | Final report (show in video) |
| `planning.md` | Design doc (already submitted via repo) |
| `data/labeled_dataset.csv` | 220 labeled posts |
| `takemeter.ipynb` | Training notebook (filled) |
| `scripts/demo.py` | Live classification for demo |
| `DEMO_SCRIPT.md` | Step-by-step narration |
| `outputs/confusion_matrix.png` | Confusion matrix image |

---

## Cursor chat tip

To find this conversation tomorrow in Cursor: open **Chat history** (clock icon in chat panel) and search for "TakeMeter" or "record video".

You don't need the chat — this file + README has everything.
