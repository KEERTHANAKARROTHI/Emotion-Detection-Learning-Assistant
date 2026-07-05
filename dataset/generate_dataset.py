"""
generate_dataset.py
--------------------
Creates dataset/emotions.csv with labeled student text samples for the
5 target emotions: Bored, Confident, Confused, Curious, Frustrated.

Run this ONCE before training:
    python dataset/generate_dataset.py

If you already have your own labeled CSV (columns: text,emotion), just
place it at dataset/emotions.csv and skip this script.
"""

import csv
import os
import random

random.seed(42)

EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

SUBJECTS = [
    "recursion", "calculus", "organic chemistry", "linear algebra",
    "machine learning", "data structures", "thermodynamics", "python loops",
    "SQL joins", "neural networks", "statistics", "grammar rules",
    "physics mechanics", "algorithms", "web development", "database design",
    "probability", "object oriented programming", "cell biology", "economics",
]

TEMPLATES = {
    "Bored": [
        "This {s} lecture is so boring, I can't focus at all.",
        "I feel so bored studying {s}, nothing seems interesting.",
        "Honestly {s} is putting me to sleep, it's just monotonous.",
        "I'm tired of {s}, it feels repetitive and dull.",
        "Studying {s} today just feels tedious and uninteresting.",
        "I keep zoning out during {s}, it's just not engaging.",
        "{s} class has become so dull, I can barely stay awake.",
    ],
    "Confident": [
        "I finally understand {s} completely, I feel really confident now.",
        "I've got {s} down, I'm sure I can solve any problem now.",
        "I feel great about {s}, everything makes sense to me.",
        "I'm confident I can ace the {s} exam after this practice.",
        "{s} used to be hard but now I feel very comfortable with it.",
        "I solved the {s} problem easily, I feel capable and sure of myself.",
        "I understand {s} well enough to explain it to someone else.",
    ],
    "Confused": [
        "I'm so lost on {s}, none of it makes sense to me.",
        "I don't understand {s} at all, I'm completely confused.",
        "I'm stuck on {s} and don't know where to even start.",
        "Can someone explain {s}? I'm really puzzled by it.",
        "I re-read the {s} chapter three times and I'm still confused.",
        "{s} concepts are unclear to me, I feel totally lost.",
        "I have no idea how {s} works, everything is a blur.",
    ],
    "Curious": [
        "I'm really curious about how {s} actually works under the hood.",
        "{s} sounds fascinating, I want to explore it further.",
        "I wonder how {s} applies to real world problems.",
        "This {s} topic is intriguing, I want to learn more about it.",
        "I'm eager to dive deeper into {s}, it seems interesting.",
        "What happens if we apply {s} differently? I'm curious to find out.",
        "I keep thinking about {s} and want to understand it deeply.",
    ],
    "Frustrated": [
        "I'm so frustrated with {s}, nothing I try works.",
        "This {s} assignment is driving me crazy, I want to give up.",
        "I've tried everything for {s} and I'm still stuck, it's so annoying.",
        "I hate how difficult {s} is, I'm getting really frustrated.",
        "Why is {s} so hard? I'm about to lose my patience.",
        "I keep making mistakes in {s} and it's so frustrating.",
        "{s} is so hard that I just want to throw my laptop.",
    ],
}


def generate_rows(per_class=120):
    rows = []
    for emotion in EMOTIONS:
        templates = TEMPLATES[emotion]
        for _ in range(per_class):
            template = random.choice(templates)
            subject = random.choice(SUBJECTS)
            text = template.format(s=subject)
            rows.append((text, emotion))
    random.shuffle(rows)
    return rows


def main():
    out_path = os.path.join(os.path.dirname(__file__), "emotions.csv")
    rows = generate_rows(per_class=120)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotion"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
