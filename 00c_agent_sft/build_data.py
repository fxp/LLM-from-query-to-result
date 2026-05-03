"""Build agent SFT traces programmatically.

Input:  hand-curated KB + arithmetic templates
Output: kb.json + data.json (used by tools.py and train.py)

Why programmatic? Because real instruction-tuning datasets are usually
synthetic or semi-synthetic — and writing 150 traces by hand is tedious
+ inconsistent. This script makes the trace structure visible and the
dataset reproducible.

Each trace is one (q, [step], answer) example. Steps follow ReAct:
  THOUGHT → ACTION → OBSERVATION (this script computes the OBSERVATION
  by actually calling the tool, so SFT data and runtime tools are in
  perfect sync).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Reuse local tools so train data uses the SAME observation strings that
# the agent loop will see at runtime. Avoids subtle "tool returns 70 vs
# observation says 70.0" mismatches.
import tools

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Knowledge base (used by lookup tool)
# ---------------------------------------------------------------------------
KB: dict[str, str] = {
    # Capitals — countries × continents to give variety
    "capital of France": "Paris",
    "capital of Germany": "Berlin",
    "capital of Italy": "Rome",
    "capital of Spain": "Madrid",
    "capital of Portugal": "Lisbon",
    "capital of UK": "London",
    "capital of England": "London",
    "capital of Ireland": "Dublin",
    "capital of Netherlands": "Amsterdam",
    "capital of Belgium": "Brussels",
    "capital of Switzerland": "Bern",
    "capital of Austria": "Vienna",
    "capital of Sweden": "Stockholm",
    "capital of Norway": "Oslo",
    "capital of Denmark": "Copenhagen",
    "capital of Finland": "Helsinki",
    "capital of Greece": "Athens",
    "capital of Poland": "Warsaw",
    "capital of Hungary": "Budapest",
    "capital of Russia": "Moscow",
    "capital of Ukraine": "Kyiv",
    "capital of Japan": "Tokyo",
    "capital of China": "Beijing",
    "capital of Korea": "Seoul",
    "capital of South Korea": "Seoul",
    "capital of India": "New Delhi",
    "capital of Vietnam": "Hanoi",
    "capital of Thailand": "Bangkok",
    "capital of Indonesia": "Jakarta",
    "capital of Singapore": "Singapore",
    "capital of Iran": "Tehran",
    "capital of Iraq": "Baghdad",
    "capital of Saudi Arabia": "Riyadh",
    "capital of Israel": "Jerusalem",
    "capital of Turkey": "Ankara",
    "capital of Egypt": "Cairo",
    "capital of Kenya": "Nairobi",
    "capital of Nigeria": "Abuja",
    "capital of South Africa": "Pretoria",
    "capital of US": "Washington",
    "capital of USA": "Washington",
    "capital of Canada": "Ottawa",
    "capital of Mexico": "Mexico City",
    "capital of Brazil": "Brasilia",
    "capital of Argentina": "Buenos Aires",
    "capital of Chile": "Santiago",
    "capital of Peru": "Lima",
    "capital of Australia": "Canberra",
    "capital of New Zealand": "Wellington",

    # Authors
    "author of Hamlet": "Shakespeare",
    "author of Macbeth": "Shakespeare",
    "author of Romeo and Juliet": "Shakespeare",
    "author of King Lear": "Shakespeare",
    "author of Othello": "Shakespeare",
    "author of The Tempest": "Shakespeare",
    "author of Pride and Prejudice": "Austen",
    "author of Emma": "Austen",
    "author of Sense and Sensibility": "Austen",
    "author of 1984": "Orwell",
    "author of Animal Farm": "Orwell",
    "author of The Great Gatsby": "Fitzgerald",
    "author of To Kill a Mockingbird": "Lee",
    "author of War and Peace": "Tolstoy",
    "author of Anna Karenina": "Tolstoy",
    "author of Crime and Punishment": "Dostoevsky",
    "author of The Brothers Karamazov": "Dostoevsky",
    "author of Don Quixote": "Cervantes",
    "author of Faust": "Goethe",

    # Painters
    "painter of Mona Lisa": "Leonardo da Vinci",
    "painter of The Last Supper": "Leonardo da Vinci",
    "painter of Starry Night": "van Gogh",
    "painter of Sunflowers": "van Gogh",
    "painter of The Scream": "Munch",
    "painter of Guernica": "Picasso",
    "painter of Sistine Chapel": "Michelangelo",

    # Chemistry
    "chemical symbol of water": "H2O",
    "chemical symbol of gold": "Au",
    "chemical symbol of silver": "Ag",
    "chemical symbol of iron": "Fe",
    "chemical symbol of oxygen": "O",
    "chemical symbol of hydrogen": "H",
    "chemical symbol of carbon": "C",
    "chemical symbol of nitrogen": "N",
    "chemical symbol of sodium": "Na",
    "chemical symbol of copper": "Cu",
    "chemical symbol of mercury": "Hg",
    "chemical symbol of lead": "Pb",

    # Geography facts
    "longest river in the world": "Nile",
    "longest river in Africa": "Nile",
    "longest river in Asia": "Yangtze",
    "longest river in Europe": "Volga",
    "longest river in South America": "Amazon",
    "tallest mountain in the world": "Mount Everest",
    "largest desert in the world": "Sahara",
    "largest ocean": "Pacific",
    "largest country": "Russia",
    "smallest country": "Vatican City",

    # Misc constants
    "speed of light": "300000 km/s",
    "boiling point of water in Celsius": "100",
    "freezing point of water in Celsius": "0",
    "days in a year": "365",
    "days in a leap year": "366",
    "days in a week": "7",
    "hours in a day": "24",
    "minutes in an hour": "60",
}


def write_kb():
    out = HERE / "kb.json"
    out.write_text(json.dumps(KB, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}: {len(KB)} entries")


# ---------------------------------------------------------------------------
# 2. Trace builder
# ---------------------------------------------------------------------------
def trace_lookup(question: str, lookup_key: str, answer_template: str) -> dict:
    """Build a 1-step lookup trace. answer_template is a format string with
    {obs} placeholder."""
    obs = tools.lookup(lookup_key)
    return {
        "q": question,
        "trace": [
            {
                "thought": f"I should look up {lookup_key}.",
                "action": f"lookup({lookup_key})",
                "observation": obs,
            }
        ],
        "answer": answer_template.format(obs=obs),
    }


def trace_calc(question: str, expr: str, answer_template: str) -> dict:
    """Build a 1-step calc trace."""
    obs = tools.calc(expr)
    return {
        "q": question,
        "trace": [
            {
                "thought": f"I need to compute {expr}.",
                "action": f"calc({expr})",
                "observation": obs,
            }
        ],
        "answer": answer_template.format(obs=obs),
    }


# ---------------------------------------------------------------------------
# 3. Generate traces
# ---------------------------------------------------------------------------
def gen_lookup_traces() -> list[dict]:
    """One trace per KB entry, with question template chosen per category."""
    out = []
    for key, val in KB.items():
        if key.startswith("capital of "):
            country = key[len("capital of "):]
            for q in (
                f"What is the capital of {country}?",
                f"What's the capital of {country}?",
            ):
                out.append(trace_lookup(q, key, "{obs}."))
        elif key.startswith("author of "):
            book = key[len("author of "):]
            for q in (f"Who wrote {book}?", f"Who is the author of {book}?"):
                out.append(trace_lookup(q, key, "{obs}."))
        elif key.startswith("painter of "):
            work = key[len("painter of "):]
            out.append(trace_lookup(f"Who painted {work}?", key, "{obs}."))
        elif key.startswith("chemical symbol of "):
            elem = key[len("chemical symbol of "):]
            out.append(trace_lookup(f"What is the chemical symbol of {elem}?",
                                     key, "{obs}."))
        elif key.startswith("longest river in "):
            place = key[len("longest river in "):]
            out.append(trace_lookup(f"What is the longest river in {place}?",
                                     key, "The {obs}."))
        else:
            # Generic "What is the <key>?"
            out.append(trace_lookup(f"What is the {key}?", key, "{obs}."))
    return out


def gen_arithmetic_traces(rng: random.Random) -> list[dict]:
    """Programmatic arithmetic: addition, subtraction, multiplication, division."""
    out = []

    # Addition: 30 examples
    for _ in range(30):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        templates = [
            f"What is {a} plus {b}?",
            f"Calculate {a} + {b}.",
            f"What's {a} + {b}?",
        ]
        out.append(trace_calc(rng.choice(templates), f"{a} + {b}", "{obs}."))

    # Subtraction: 15 examples
    for _ in range(15):
        a = rng.randint(50, 200)
        b = rng.randint(1, a - 1)
        out.append(trace_calc(
            rng.choice([f"What is {a} minus {b}?", f"What is {a} - {b}?"]),
            f"{a} - {b}", "{obs}."))

    # Multiplication: 20 examples
    for _ in range(20):
        a, b = rng.randint(2, 25), rng.randint(2, 25)
        out.append(trace_calc(
            rng.choice([f"What is {a} times {b}?", f"What is {a} * {b}?"]),
            f"{a} * {b}", "{obs}."))

    # Division (clean): 10 examples
    for _ in range(10):
        b = rng.choice([2, 3, 4, 5, 6, 8, 10])
        a = b * rng.randint(2, 20)
        out.append(trace_calc(f"What is {a} divided by {b}?",
                               f"{a} / {b}", "{obs}."))

    # Mixed expressions: 10 examples
    for _ in range(10):
        a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 10)
        op1 = rng.choice(["+", "-"])
        out.append(trace_calc(f"What is ({a} {op1} {b}) * {c}?",
                               f"({a} {op1} {b}) * {c}", "{obs}."))

    return out


def write_data():
    rng = random.Random(42)
    traces = gen_lookup_traces() + gen_arithmetic_traces(rng)
    rng.shuffle(traces)

    out = HERE / "data.json"
    out.write_text(json.dumps(traces, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    n_lookup = sum(1 for t in traces if t["trace"][0]["action"].startswith("lookup"))
    n_calc = sum(1 for t in traces if t["trace"][0]["action"].startswith("calc"))
    avg_len = sum(len(t["q"]) + len(t["answer"])
                  + sum(len(s["thought"]) + len(s["action"]) + len(s["observation"])
                        for s in t["trace"]) for t in traces) / len(traces)
    print(f"wrote {out}: {len(traces)} traces  "
          f"({n_lookup} lookup + {n_calc} calc), avg ~{avg_len:.0f} chars")


if __name__ == "__main__":
    write_kb()
    write_data()
