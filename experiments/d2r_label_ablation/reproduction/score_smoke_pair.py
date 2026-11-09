"""Score frozen smoke validation answers and compare actions and replayed costs."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import string

VARIANTS = ("baseline", "with_label", "without_label")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(answer):
    if not isinstance(answer, str):
        raise TypeError("Answers must be strings.")
    text = "".join(
        char for char in answer.lower() if char not in string.punctuation
    )
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, gold):
    predicted, expected = normalize(prediction), normalize(gold)
    exact = float(predicted == expected)
    special = {"yes", "no", "noanswer"}
    if (predicted in special or expected in special) and predicted != expected:
        f1 = 0.0
    else:
        left, right = predicted.split(), expected.split()
        overlap = sum((Counter(left) & Counter(right)).values())
        f1 = 2.0 * overlap / (len(left) + len(right)) if overlap else 0.0
    phrase = bool(expected) and f" {expected} " in f" {predicted} "
    return {"em": exact, "f1": f1, "gold_phrase_present": phrase}


def cost_gap(left, right):
    midpoint = (left + right) / 2
    return 100 * abs(left - right) / midpoint if midpoint else 0.0


def summarize(records):
    result = {}
    for name in VARIANTS:
        entries = [row["scores"][name] for row in records]
        result[name] = {
            "n": len(entries),
            "em_percent": 100 * mean(row["em"] for row in entries),
            "f1_percent": 100 * mean(row["f1"] for row in entries),
            "mean_replayed_seconds": mean(row["seconds"] for row in entries),
            "mean_input_tokens": mean(row["input_tokens"] for row in entries),
            "mean_output_tokens": mean(row["output_tokens"] for row in entries),
            "longer_answers_containing_gold": sum(
                row["gold_phrase_present"] and row["em"] == 0
                for row in entries
            ),
            "actions": dict(Counter(
                "keep" if row["action_id"] is None else str(row["action_id"])
                for row in entries
            )),
        }
    result["label_cost_gap_percent"] = cost_gap(
        result["with_label"]["mean_replayed_seconds"],
        result["without_label"]["mean_replayed_seconds"],
    )
    for name in ("with_label", "without_label"):
        result[name]["em_fixed"] = sum(
            row["scores"]["baseline"]["em"] == 0
            and row["scores"][name]["em"] == 1
            for row in records
        )
        result[name]["em_broken"] = sum(
            row["scores"]["baseline"]["em"] == 1
            and row["scores"][name]["em"] == 0
            for row in records
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("draft_run", type=Path)
    parser.add_argument("policy_run", type=Path)
    args = parser.parse_args()
    prepared, draft_run, policy_run = (
        path.resolve()
        for path in (args.prepared, args.draft_run, args.policy_run)
    )
    draft_file = draft_run / "controlled_drafts.jsonl"
    split_file = draft_run / "smoke_split.json"
    journal = json.loads((policy_run / "journal.json").read_text())
    split = json.loads(split_file.read_text())
    if hashlib.sha256(draft_file.read_bytes()).hexdigest() != journal["config"]["drafts"]:
        raise SystemExit("Draft file differs from the training configuration.")
    if hashlib.sha256(split_file.read_bytes()).hexdigest() != journal["config"]["split"]:
        raise SystemExit("Split differs from the training configuration.")

    drafts = {row["case_id"]: row for row in read_rows(draft_file)}
    audit = {
        row["case_id"]: row
        for row in read_rows(prepared / "audit.jsonl")
    }
    predictions = read_rows(policy_run / "validation.jsonl")
    expected = {
        cid for cid in drafts
        if audit[cid]["question_id"] in split["validation_question_ids"]
    }
    if (len(predictions) != len(expected)
            or {row["case_id"] for row in predictions} != expected):
        raise SystemExit("Predictions do not match the frozen validation cases.")
    if predictions != journal["validation"]:
        raise SystemExit("Validation export differs from the saved journal.")

    scored = []
    action_agreements = answer_agreements = 0
    for row in predictions:
        cid = row["case_id"]
        meta = audit[cid]
        if (row["question_id"] != meta["question_id"]
                or row["condition"] != meta["condition"]):
            raise SystemExit(f"Validation metadata differs for {cid}")
        gold = meta["gold_answer"]
        baseline = drafts[cid]
        entries = {
            "baseline": {
                **answer_scores(baseline["response"], gold),
                "answer": baseline["response"],
                "action_id": None,
                "seconds": baseline["end_to_end_latency_s"],
                "input_tokens": baseline["input_tokens"],
                "output_tokens": baseline["output_tokens"],
            }
        }
        for name in ("with_label", "without_label"):
            decision = row["variants"][name]
            entries[name] = {
                **answer_scores(decision["answer"], gold),
                "answer": decision["answer"],
                "action_id": decision["action_id"],
                "seconds": decision["replayed_seconds"],
                "input_tokens": decision["input_tokens"],
                "output_tokens": decision["output_tokens"],
            }
        left, right = (
            row["variants"][name]
            for name in ("with_label", "without_label")
        )
        action_agreements += left["action_id"] == right["action_id"]
        answer_agreements += left["answer"] == right["answer"]
        if left["action_id"] == right["action_id"]:
            if (left["outcome_key"] != right["outcome_key"]
                    or left["answer"] != right["answer"]):
                raise RuntimeError("Equal actions did not share the recorded outcome.")
        if any(
            not math.isfinite(entry["seconds"]) or entry["seconds"] < 0
            for entry in entries.values()
        ):
            raise RuntimeError("Invalid replayed time.")
        scored.append({
            "case_id": cid,
            "question_id": meta["question_id"],
            "condition": meta["condition"],
            "gold_answer": gold,
            "scores": entries,
        })

    groups = {"overall": summarize(scored)}
    for condition in sorted({row["condition"] for row in scored}):
        groups[condition] = summarize([
            row for row in scored if row["condition"] == condition
        ])
    summary = {
        "validation_cases": len(scored),
        "validation_questions": len({row["question_id"] for row in scored}),
        "action_agreements": action_agreements,
        "identical_answers": answer_agreements,
        "groups": groups,
    }
    outputs = (
        ("scored_validation.jsonl", "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in scored
        )),
        ("score_summary.json",
         json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
    )
    for name, content in outputs:
        target = policy_run / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(content)
        os.replace(temporary, target)

    print("Validation cases:", summary["validation_cases"],
          "| original questions:", summary["validation_questions"])
    print("Action agreements:", action_agreements, "/", len(scored))
    print("Identical final answers:", answer_agreements, "/", len(scored))
    for group, metrics in groups.items():
        print("\n" + group)
        print("variant        n    EM%    F1%    replayed_s")
        for name in VARIANTS:
            item = metrics[name]
            print(
                f"{name:<14} {item['n']:>2} "
                f"{item['em_percent']:>6.2f} {item['f1_percent']:>6.2f} "
                f"{item['mean_replayed_seconds']:>11.4f}"
            )
        print(f"Label cost gap: {metrics['label_cost_gap_percent']:.3f}%")
    for name in ("with_label", "without_label"):
        item = groups["overall"][name]
        print(name, "| actions:", item["actions"],
              "| longer answers containing gold:",
              item["longer_answers_containing_gold"])
    print("Saved:", policy_run / "score_summary.json")


if __name__ == "__main__":
    main()
