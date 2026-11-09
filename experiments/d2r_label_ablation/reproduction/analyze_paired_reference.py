import hashlib
import json
from pathlib import Path
import random
from statistics import mean, quantiles

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "policy_run", nargs="?",
    type=lambda s: Path(s).expanduser().resolve(),
    default=Path.home() / "dsc180-runs/label-policy-pilot-01",
)
args = parser.parse_args()
folder = args.policy_run / "offline-correctness-01"
source = folder / "reference_scores.jsonl"
rows = [
    json.loads(line)
    for line in source.read_text().splitlines()
    if line.strip()
]

cases = {}
for row in rows:
    item = cases.setdefault(row["case_id"], {
        "question_id": row["question_id"],
        "condition": row["condition"],
        "variants": {},
    })
    assert item["question_id"] == row["question_id"]
    assert item["condition"] == row["condition"]
    assert row["variant"] not in item["variants"]
    assert type(row["correct"]) is bool
    item["variants"][row["variant"]] = row["correct"]

assert len(cases) == 48
assert all(
    set(item["variants"]) == {
        "baseline", "with_label", "without_label"
    }
    for item in cases.values()
)

by_question = {}
for item in cases.values():
    by_question.setdefault(
        item["question_id"], []
    ).append(item["condition"])

expected = {
    "clean_complete", "missing_support", "complete_distractors"
}
assert len(by_question) == 16
assert all(
    len(conditions) == 3 and set(conditions) == expected
    for conditions in by_question.values()
)

groups = {"overall": list(cases.values())}
for condition in sorted(expected):
    groups[condition] = [
        item for item in cases.values()
        if item["condition"] == condition
    ]

comparisons = [
    ("with_label", "without_label"),
    ("with_label", "baseline"),
    ("without_label", "baseline"),
]
result = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "method": "paired bootstrap resampling original questions",
    "bootstrap_replicates": 10000,
    "seed": 180,
    "groups": {},
}

for group, members in groups.items():
    qids = sorted({item["question_id"] for item in members})
    rng = random.Random(180)
    draws = [
        rng.choices(qids, k=len(qids))
        for _ in range(10000)
    ]
    result["groups"][group] = {}
    print("\n" + group)

    for a, b in comparisons:
        differences = {qid: [] for qid in qids}
        wins = losses = 0
        for item in members:
            va = item["variants"][a]
            vb = item["variants"][b]
            wins += int(va and not vb)
            losses += int(vb and not va)
            differences[item["question_id"]].append(
                int(va) - int(vb)
            )

        assert len({
            len(values) for values in differences.values()
        }) == 1
        qmeans = {
            qid: mean(values)
            for qid, values in differences.items()
        }
        samples = [
            mean(qmeans[qid] for qid in draw)
            for draw in draws
        ]
        cuts = quantiles(samples, n=40, method="inclusive")
        entry = {
            "cases": len(members),
            "original_questions": len(qids),
            "a_correct_b_wrong": wins,
            "a_wrong_b_correct": losses,
            "accuracy_difference_pp":
                100 * (wins - losses) / len(members),
            "bootstrap_95_ci_pp": [
                100 * cuts[0], 100 * cuts[-1]
            ],
        }
        name = a + " minus " + b
        result["groups"][group][name] = entry
        print(name, json.dumps(entry))

target = folder / "paired_reference_analysis.json"
target.write_text(json.dumps(result, indent=2) + "\n")
print("\nSaved:", target)
