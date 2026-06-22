#!/usr/bin/env python3
"""Gradio web demo for TakeMeter (stretch feature)."""

from __future__ import annotations

import gradio as gr
from demo import classify, load_model

tokenizer, model = load_model()


def predict(text: str):
    if not text.strip():
        return "—", 0.0
    label, conf = classify(text, tokenizer, model)
    return label, conf


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="r/nba post or comment", lines=4, placeholder="Paste a post..."),
    outputs=[
        gr.Textbox(label="Predicted label"),
        gr.Number(label="Confidence", precision=2),
    ],
    title="TakeMeter — r/nba Discourse Classifier",
    description="Classifies posts as **analysis**, **hot_take**, or **reaction**.",
    examples=[
        ["Denver's drop coverage leaves the corner three open — they rank 28th in opponent corner 3PA rate."],
        ["Trade Luka now before his attitude destroys the franchise. Mark my words."],
        ["WHAT A BLOCK. I literally jumped off my couch."],
    ],
)

if __name__ == "__main__":
    demo.launch()
