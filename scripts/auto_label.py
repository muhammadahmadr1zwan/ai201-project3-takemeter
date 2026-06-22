#!/usr/bin/env python3
"""
Apply TakeMeter label taxonomy to raw_posts.csv.
Uses rule-based classification aligned with planning.md definitions.
Review output before training — treat as pre-labeling assistance.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_posts.csv"
OUT = ROOT / "data" / "labeled_dataset.csv"

# Non-discourse / spam — exclude from dataset
SKIP_PATTERNS = [
    r"wireless earbuds",
    r"iptv",
    r"school's out for summer",
    r"sars-cov",
    r"just dropped this.*tee",
    r"^smart wireless",
]

REACTION_PATTERNS = [
    r"\blmao\b",
    r"\bwild\b",
    r"i literally jumped",
    r"i can't do this",
    r"what a block",
    r"congrats okc",
    r"hope this painting",
    r"🔥",
    r"meatless shai",
    r"wolves in \d",
    r"warriors fans crying",
    r"nba betting fixed",
    r"so damn proud",
    r"the game was unwatchable",
    r"i had to turn it off",
    r"ratio'd",
    r"might be the craziest take",
    r"^rondo was the goat",
    r"^jokic is still the best",
    r"^celtics should have won",
    r"^nuggets lost because",
    r"^russell westbrook is a basketball liability",
    r"^jokic isn't a top 5",
    r"^the nba should be thankful okc",
    r"^20\+ years we will find out",
    r"^remove scott foster",
    r"^vince carter dunk",
    r"^shai via ig",
    r"^there was a water leak",
    r"^lmao this might be",
    r"^nba has to get their shit straight",  # venting tone but has argument - borderline
]

HOT_TAKE_PATTERNS = [
    r"\boverrated\b",
    r"\bfraud\b",
    r"\bfrauds\b",
    r"mark my words",
    r"trade .+ for",
    r"should (the|we) trade",
    r"is .+ the best player",
    r"is .+ the goat",
    r"can .+ be the .?face of",
    r"who ya got",
    r"what do you think\?",
    r"change my view",
    r"\bcmv\b",
    r"hot take",
    r"unpopular opinion",
    r"prove that",
    r"is it crazy to say",
    r"is luka an underachiever",
    r"is luka doncic",
    r"will joker be asking",
    r"should the nuggets trade",
    r"only a handful of rings should count",
    r"is it generally agreed",
    r"does jokic need to",
    r"if shai wins mvp.*best player",
    r"has giannis proven",
    r"where does jokic.*rate all time",
    r"would you rather",
    r"who is the better",
    r"who's your pick",
    r"who are you rooting",
    r"which team are you rooting",
    r"down to the final four",
    r"basically the options",
    r"is sga already",
    r"is shai a playoff",
    r"do you think it is good or bad",
    r"why did the dallas cowboys",
    r"is luke docick overrated",
    r"why did steph curry have the injury",
    r"i am going to start watching",
    r"what's your favorite memory",
    r"what's better.*posterizing",
    r"what is better.*posterizing",
    r"did anyone else collect",
    r"whose individual performance",
    r"was sga laughing.*coldest moments",
    r"was steph curry dumb",
    r"why was a possible tim donaghy",
    r"when is nba mvp gonna be announced",
]

ANALYSIS_PATTERNS = [
    r"net rating",
    r"\+?\d+\.\d+%",
    r"\|\s*\+?\-?\d+",
    r"ppg on \d",
    r"per 36",
    r"since \d{4}",
    r"in the last \d+",
    r"over the last \d+",
    r"histor(y|ical)",
    r"stathead",
    r"source:",
    r"\|\s*rank\s*\|",
    r"ortg|drtg|ts%",
    r"turnover",
    r"free throw",
    r"championship drought",
    r"conference finals appearance",
    r"remaining teams by salary",
    r"collective bargaining",
    r"basketball related income",
    r"stepien rule",
    r"second apron",
    r"sign and trade",
    r"playoff series record",
    r"point differential",
    r"unique champions",
    r"different champions in \d+",
    r"steals in this series",
    r"starters dominant playoff",
    r"how to win in the playoffs",
    r"guards who play with energy",
    r"referee behaviors",
    r"scouting report.*referee",
    r"assist should be credited",
    r"schedule.*rest between",
    r"expansion franchises",
    r"82 games",
    r"70-game season",
    r"commercials.*replay challenges",
    r"drop coverage",
    r"flagrant.*reviewed",
    r"mpj after the asb",
    r"sga and jdub finish",
    r"best players.*net points",
    r"westbrook against the thunder.*netrtg",
    r"jokic attempted more ft",
    r"free throws attempted in the \d{4}",
    r"only \d+ players remaining",
    r"3/5 of the knicks starters",
    r"for the first time since \d{4}",
    r"2011-25 will have \d+ unique",
    r"in 21st century.*50-win",
    r"denver nuggets have extraordinarily limited",
    r"why the nba will never reduce",
    r"why the wcf starts before",
]


def should_skip(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in SKIP_PATTERNS)


def count_stats(text: str) -> int:
    """Rough count of statistical/evidence markers."""
    patterns = [
        r"\d+\.\d+",
        r"\d+%",
        r"\d+-\d+",
        r"\+\d+",
        r"\-\d+",
        r"\b\d{4}\b",
        r"\|\s",
        r"ppg|rpg|apg|ts%|fg%|3pt",
        r"game \d",
        r"series",
    ]
    return sum(len(re.findall(p, text, re.I)) for p in patterns)


def classify(text: str) -> tuple[str, str]:
    lower = text.lower()

    for p in REACTION_PATTERNS:
        if re.search(p, lower):
            return "reaction", "matched reaction pattern"

    stat_count = count_stats(text)
    has_table = "|" in text and text.count("|") >= 4
    has_structured_list = bool(re.search(r"(\d+\.|•|\*)\s", text))

    analysis_hits = sum(1 for p in ANALYSIS_PATTERNS if re.search(p, lower))
    hot_hits = sum(1 for p in HOT_TAKE_PATTERNS if re.search(p, lower))

    # Strong analysis signals
    if has_table or (stat_count >= 8 and analysis_hits >= 1):
        return "analysis", "stats/table heavy"
    if analysis_hits >= 2 and stat_count >= 4:
        return "analysis", "structured evidence"
    if stat_count >= 6 and len(text) > 200 and hot_hits == 0:
        return "analysis", "stat-dense post"

    # Single-stat accusatory -> hot_take per edge rule
    if stat_count <= 3 and hot_hits >= 1:
        return "hot_take", "opinion/question pattern"
    if re.search(r"overrated|fraud|frauds|trade .+ for|should trade", lower):
        return "hot_take", "bold claim"
    # Debate questions are hot_take, not reaction
    if text.endswith("?") and stat_count < 4 and not has_table:
        if re.search(r"predictions|who (ya got|you got|are you rooting|do you think)", lower):
            return "hot_take", "prediction/debate question"
        return "hot_take", "debate question without evidence"

    # Video/highlight link titles without emotion -> hot_take
    if re.search(r"^\[(highlight|gil's arena|bill simmons)\]", lower):
        return "hot_take", "link post title"

    # Quote posts / short emotional
    if re.search(r"^(jokic:|shai |chet on|aaron gordon on|alex caruso on|jamal murray)", lower):
        if stat_count >= 3:
            return "analysis", "stat quote"
        return "reaction", "quote/commentary"

    if re.search(r"^(congrats|nba betting|meatless|warriors fans|russell westbrook is|celtics should)", lower):
        return "reaction", "emotional vent"

    # News/reporting with stats -> analysis
    if stat_count >= 5 and len(text) > 150:
        return "analysis", "stat reporting"

    # Default hot_take for opinion posts, reaction for very short
    if len(text) < 100:
        return "reaction", "short default"
    if hot_hits >= 1:
        return "hot_take", "hot_take default"
    if stat_count >= 3:
        return "analysis", "moderate stats"

    return "hot_take", "fallback opinion discourse"


def main() -> None:
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    labeled = []

    for row in rows:
        text = row["text"]
        if should_skip(text):
            continue
        label, reason = classify(text)
        labeled.append({"text": text, "label": label, "notes": reason})

    # Balance: cap each label at 75, min 220 total
    from collections import defaultdict

    by_label: dict[str, list] = defaultdict(list)
    for item in labeled:
        by_label[item["label"]].append(item)

    target_per = 73
    balanced = []
    for label in ["analysis", "hot_take", "reaction"]:
        pool = by_label[label]
        balanced.extend(pool[:target_per])

    # Fill to 220 if short
    used = {id(x) for x in balanced}
    remaining = [x for x in labeled if id(x) not in used]
    while len(balanced) < 220 and remaining:
        balanced.append(remaining.pop(0))

    balanced = balanced[:220]

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
        writer.writeheader()
        writer.writerows(balanced)

    from collections import Counter

    counts = Counter(r["label"] for r in balanced)
    print(f"Wrote {len(balanced)} rows to {OUT}")
    for label, n in counts.most_common():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
