"""Export blinded, deduplicated evaluation inputs from cached smoke outputs."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sqlite3


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def save(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("draft_run", type=Path)
    parser.add_argument("policy_run", type=Path)
    parser.add_argument("shared_repair", type=Path)
    args = parser.parse_args()
    prepared, draft_run, policy_run, shared = (
        path.resolve() for path in
        (args.prepared, args.draft_run, args.policy_run, args.shared_repair)
    )
    draft_file = draft_run / "controlled_drafts.jsonl"
    drafts = {row["case_id"]: row for row in rows(draft_file)}
    audit = {row["case_id"]: row for row in rows(prepared / "audit.jsonl")}
    decisions = rows(policy_run / "validation.jsonl")
    journal = json.loads((policy_run / "journal.json").read_text())
    if decisions != journal["validation"]:
        raise SystemExit("Validation file differs from the journal.")
    if hashlib.sha256(draft_file.read_bytes()).hexdigest() != journal["config"]["drafts"]:
        raise SystemExit("Draft file differs from the training configuration.")
    database_path = shared / "rollouts.sqlite3"
    if not database_path.is_file():
        raise SystemExit("Repair cache is missing.")

    inputs, mapping = {}, []
    with sqlite3.connect(database_path) as database:
        for decision in decisions:
            cid = decision["case_id"]
            draft = drafts[cid]
            for variant in ("baseline", "with_label", "without_label"):
                outcome = draft
                if variant != "baseline":
                    route = decision["variants"][variant]
                    key = route["outcome_key"]
                    if key is not None:
                        cached = database.execute(
                            "SELECT result FROM repairs WHERE key = ?", (key,)
                        ).fetchone()
                        if cached is None:
                            raise SystemExit(f"Missing cached repair: {key}")
                        outcome = json.loads(cached[0])["response"]
                    if outcome["response"] != route["answer"]:
                        raise SystemExit(f"Cached answer differs for {cid} / {variant}")

                contexts = [
                    node["text"] for node in outcome["retrieved_context"]
                ]
                if not contexts or any(
                    not isinstance(text, str) for text in contexts
                ):
                    raise SystemExit(f"Missing generation evidence for {cid} / {variant}")
                item = {
                    "question": draft["question"],
                    "answer": outcome["response"],
                    "contexts": contexts,
                    "ground_truth": audit[cid]["gold_answer"],
                }
                evaluation_id = hashlib.sha256(encoded(item).encode()).hexdigest()
                inputs[evaluation_id] = {
                    "evaluation_id": evaluation_id, **item
                }
                mapping.append({
                    "case_id": cid,
                    "question_id": audit[cid]["question_id"],
                    "condition": audit[cid]["condition"],
                    "variant": variant,
                    "evaluation_id": evaluation_id,
                })

    versions = {}
    for package in (
        "ragas", "langchain-core", "llama-index-core",
        "llama-index-llms-openai", "openai",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "MISSING"

    save(policy_run / "evaluation_inputs.jsonl", "".join(
        encoded(item) + "\n" for item in inputs.values()
    ))
    save(policy_run / "evaluation_mapping.jsonl", "".join(
        encoded(item) + "\n" for item in mapping
    ))
    save(policy_run / "evaluation_export.json", encoded({
        "policy_config": journal["config"],
        "versions": versions,
        "unique_evaluations": len(inputs),
        "comparison_rows": len(mapping),
        "validation_sha256": hashlib.sha256(
            (policy_run / "validation.jsonl").read_bytes()
        ).hexdigest(),
        "evaluation_inputs_sha256": hashlib.sha256(
            (policy_run / "evaluation_inputs.jsonl").read_bytes()
        ).hexdigest(),
    }) + "\n")

    print("Comparison rows:", len(mapping))
    print("Unique answer-and-evidence inputs:", len(inputs))
    print("Versions:", encoded(versions))
    print("Saved:", policy_run / "evaluation_inputs.jsonl")
    print("No API calls made.")


if __name__ == "__main__":
    main()
