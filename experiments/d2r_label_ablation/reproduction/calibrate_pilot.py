"""Calibrate pilot budgets on one flagged training case using all upstream actions."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
from controlled_repair import ACTION_UPDATES, BASE_PARAMS, digest, encode
from label_features import repair_required

LAUNCHER = """
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "reproduction"))
import controlled_repair
sys.argv = ["controlled_repair.py", *sys.argv[1:]]
try:
    controlled_repair.main()
except Exception as error:
    print("\\nError chain:", flush=True)
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        print(type(error).__name__, str(error), flush=True)
        error = error.__cause__ or error.__context__
    raise SystemExit(1)
"""


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cached_result(database_path, key):
    with sqlite3.connect(database_path) as database:
        row = database.execute(
            "SELECT result FROM repairs WHERE key = ?", (key,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(encode(value) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("draft_run", type=Path)
    parser.add_argument("shared_repair", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=66)
    parser.add_argument("--requests-per-action", type=int, default=6)
    args = parser.parse_args()
    prepared, draft_run, shared = (
        path.resolve() for path in
        (args.prepared, args.draft_run, args.shared_repair)
    )
    if args.max_new_requests <= 0 or args.requests_per_action <= 0:
        parser.error("Request allowances must be positive.")

    repo = Path(__file__).resolve().parents[1]
    executor = repo / "reproduction/controlled_repair.py"
    draft_file = draft_run / "controlled_drafts.jsonl"
    split_file = draft_run / "pilot_split.json"
    split = json.loads(split_file.read_text())
    if digest(draft_file.read_bytes()) != split["drafts_sha256"]:
        raise SystemExit("Drafts differ from the frozen pilot split.")

    case_id = split["calibration_case_id"]
    draft = next(row for row in rows(draft_file) if row["case_id"] == case_id)
    audit = next(row for row in rows(prepared / "audit.jsonl")
                 if row["case_id"] == case_id)
    if (audit["question_id"] not in split["train_question_ids"]
            or not repair_required(draft)):
        raise SystemExit("Calibration requires a flagged training case.")

    source = repo / "D2R-RAG/src"
    secondary = prepared / "repair_secondary.jsonl"
    signature = {
        "corpus": digest((prepared / "repair_corpus.jsonl").read_bytes()),
        "secondary": digest(secondary.read_bytes()),
        "source": {
            path.name: digest(path.read_bytes())
            for path in sorted(source.iterdir())
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
        },
    }
    identities = [{
        "case_id": case_id,
        "action_id": action_id,
        "kg_enabled": True,
        "kb_seed": 180,
        "params": dict(BASE_PARAMS, **updates),
        "draft_sha256": digest(encode(draft).encode()),
        "index": signature,
        "executor_sha256": digest(executor.read_bytes()),
    } for action_id, updates in enumerate(ACTION_UPDATES)]
    keys = [digest(encode(identity).encode()) for identity in identities]
    database_path = shared / "rollouts.sqlite3"
    if not database_path.is_file():
        raise SystemExit("The shared repair cache does not exist.")

    results = [cached_result(database_path, key) for key in keys]
    print("Calibration case:", case_id, flush=True)
    print("Cached actions:", sum(result is not None for result in results),
          "/ 11", flush=True)
    if not args.run:
        print("Preview complete. Add --run to collect missing outcomes.")
        return

    import torch
    runtime = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version,
    }
    plan_inputs = {
        "split_sha256": digest(split_file.read_bytes()),
        "identities": identities,
        "runtime": runtime,
    }
    ledger = repo / "results/smoke-budget.json"
    state = json.loads(ledger.read_text())
    plan_file = draft_run / "calibration_plan.json"
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        if plan["inputs"] != plan_inputs:
            raise SystemExit("Calibration inputs or runtime differ from the saved plan.")
    else:
        plan = {
            "inputs": plan_inputs,
            "requests_at_start": state["requests"],
            "max_total_requests": state["requests"] + args.max_new_requests,
        }
        save_json(plan_file, plan)

    print("Cumulative request cap:", plan["max_total_requests"], flush=True)
    log_dir = draft_run / "calibration-logs"
    log_dir.mkdir(exist_ok=True)

    for action_id, result in enumerate(results):
        if result is None:
            log_file = log_dir / f"action-{action_id}.log"
            command = [
                sys.executable, "-u", "-c", LAUNCHER,
                str(prepared), str(draft_file), str(shared),
                "--case-id", case_id,
                "--action-id", str(action_id),
                "--with-kg", "--kb-seed", "180",
                "--max-new-requests", str(args.requests_per_action),
                "--max-total-requests", str(plan["max_total_requests"]),
            ]
            print(f"RUN action {action_id}; log: {log_file}", flush=True)
            with log_file.open("a", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT
                )
            if completed.returncode:
                print("\n".join(log_file.read_text().splitlines()[-25:]))
                raise SystemExit(
                    f"Action {action_id} failed. Rerun this batch to resume."
                )
            result = cached_result(database_path, keys[action_id])
            if result is None:
                raise RuntimeError(f"Action {action_id} has no expected cache entry.")
            results[action_id] = result
        print(f"SAVED action {action_id}", flush=True)

    measured = []
    for action_id, result in enumerate(results):
        response = result["response"]
        if "consistency_check" not in response or "entailment_check" not in response:
            raise RuntimeError(f"Action {action_id} has incomplete diagnostics.")
        measured.append({
            "action_id": action_id,
            "params": identities[action_id]["params"],
            "latency_s": float(response["latency"]),
            "vram_mb": float(response["vram_usage"]),
            "wall_seconds": result["wall_seconds"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "kg_consistency": response["consistency_check"],
            "kb_changed": result["kb_changed"],
            "chat_log": result["chat_log"],
        })

    latencies = np.array([row["latency_s"] for row in measured])
    memories = np.array([row["vram_mb"] for row in measured])
    if not np.all(np.isfinite(latencies)) or not np.all(np.isfinite(memories)):
        raise RuntimeError("Non-finite resource measurements.")
    if np.any(latencies <= 0) or np.any(memories < 0):
        raise RuntimeError("Invalid latency or VRAM measurement.")
    budgets = {
        "latency_s": max(0.001, float(np.percentile(latencies, 90))),
        "vram_mb": max(1.0, float(np.percentile(memories, 90))),
    }
    target = draft_run / "action_calibration.json"
    save_json(target, {
        "scope": "single-training-case pilot calibration",
        "case_id": case_id,
        "base_budgets": budgets,
        "budget_scales": [0.7, 1.0, 1.5],
        "actions": measured,
        "plan": str(plan_file),
    })
    print("Base budgets:", encode(budgets))
    print("Saved:", target)


if __name__ == "__main__":
    main()
