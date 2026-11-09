"""Review saved answers without displaying policy or judge verdicts."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "folder", type=lambda value: Path(value).expanduser().resolve()
)
args = parser.parse_args()
path = args.folder / "manual_review.jsonl"
items = [
    json.loads(line)
    for line in path.read_text().splitlines()
    if line.strip()
]

def save():
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(
        json.dumps(item, ensure_ascii=False) + "\n"
        for item in items
    ))
    os.replace(temporary, path)

for index, item in enumerate(items, 1):
    if type(item.get("human_correct")) is bool:
        continue
    print("\nItem", index, "/", len(items))
    print("Question:", item["question"])
    print("Reference:", item["reference_answer"])
    print("Candidate:", item["candidate_answer"])

    while True:
        answer = input(
            "Correct answer? y=yes, n=no, u=unsure, q=save and quit: "
        ).strip().lower()
        if answer in {"y", "n", "u", "q"}:
            break

    if answer == "q":
        break
    if answer == "u":
        item["notes"] = (
            item.get("notes") or "Uncertain; needs adjudication."
        )
    else:
        item["human_correct"] = answer == "y"
    save()

save()
judgments = {}
for line in (args.folder / "reference_scores.jsonl").read_text().splitlines():
    if line.strip():
        row = json.loads(line)
        if row["judge_id"] in judgments:
            assert judgments[row["judge_id"]] == row["correct"]
        judgments[row["judge_id"]] = row["correct"]

reviewed = [
    item for item in items
    if type(item.get("human_correct")) is bool
]
disagreements = [
    {
        "judge_id": item["judge_id"],
        "human_correct": item["human_correct"],
        "judge_correct": judgments[item["judge_id"]],
    }
    for item in reviewed
    if item["human_correct"] != judgments[item["judge_id"]]
]
result = {
    "reviewed": len(reviewed),
    "sample_size": len(items),
    "agreements": len(reviewed) - len(disagreements),
    "disagreements": disagreements,
}
(args.folder / "manual_review_results.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("\nReview summary:", json.dumps(result, indent=2))
