# Demo Video Script (~3–5 min)

Use this while recording. Show terminal + README evaluation section.

## 1. Intro (30 sec)

- "TakeMeter classifies r/nba posts as analysis, hot_take, or reaction."
- Open README → Label Taxonomy table

## 2. Classify 3–5 posts (90 sec)

Run in terminal (label + confidence print):

```powershell
python scripts/demo.py "Denver ranks 28th in opponent corner 3PA rate. Teams with elite pull-up shooters exploit drop coverage."
python scripts/demo.py "Trade Luka now before his attitude destroys the franchise. Mark my words."
python scripts/demo.py "WHAT A BLOCK. I literally jumped off my couch."
python scripts/demo.py "SGA and Jdub finish the series with more assists than Jokic. SGA: 46 assists, Jokic: 41."
python scripts/demo.py "Conference Finals Starting Lineups Ranked by Net Rating with full table stats."
```

## 3. One CORRECT prediction — narrate (45 sec)

Pick the **hot_take** or **analysis** example that matches.

Say: "This is hot_take because it asserts without building an argument — no structured evidence, just a bold claim."

## 4. One WRONG prediction — narrate (45 sec)

Use the **SGA assists** example if model predicts analysis.

Say: "True label is hot_take — it's a stat selected for effect. The model saw numbers and predicted analysis. This matches our confusion matrix: analysis and hot_take get confused when stats appear."

## 5. Evaluation walkthrough (60 sec)

Scroll README to:
- Overall accuracy table (42.4% baseline vs 51.5% fine-tuned)
- Confusion matrix — point out reaction row predicted as hot_take
- Reflection paragraph

## 6. Close (15 sec)

- "Fine-tuning helped but reaction class needs more data and clearer boundaries."
