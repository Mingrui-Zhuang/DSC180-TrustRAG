"""Train paired LinUCB smoke policies and record frozen validation decisions."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from types import SimpleNamespace

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
from calibrate_smoke import LAUNCHER, cached_result, rows, save_json
from controlled_repair import digest, encode
from label_features import paired_contexts, repair_required

NAMES = ("with_label", "without_label")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("draft_run", type=Path)
    parser.add_argument("shared_repair", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-new-requests", type=int, default=120)
    parser.add_argument("--requests-per-action", type=int, default=6)
    args = parser.parse_args()
    if not np.isfinite(args.budget_scale) or args.budget_scale <= 0:
        parser.error("Budget scale must be positive and finite.")
    if min(args.epochs, args.max_new_requests, args.requests_per_action) <= 0:
        parser.error("Epochs and request limits must be positive.")
    if args.run and not os.environ.get("OPENAI_API_KEY"):
        parser.error("Load OPENAI_API_KEY in this shell before --run.")

    prepared, draft_run, shared, output = (
        path.resolve() for path in
        (args.prepared, args.draft_run, args.shared_repair, args.output)
    )
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "D2R-RAG"))
    draft_file = draft_run / "controlled_drafts.jsonl"
    split_file = draft_run / "smoke_split.json"
    calibration_file = draft_run / "action_calibration.json"
    split = json.loads(split_file.read_text())
    calibration = json.loads(calibration_file.read_text())
    plan = json.loads((draft_run / "calibration_plan.json").read_text())
    templates = plan["inputs"]["identities"]
    executor = repo / "reproduction/controlled_repair.py"

    if digest(draft_file.read_bytes()) != split["drafts_sha256"]:
        raise SystemExit("Drafts differ from the frozen split.")
    if len(calibration["actions"]) != 11 or len(templates) != 11:
        raise SystemExit("Complete the 11-action calibration first.")
    if any(
        template["action_id"] != index
        or template["executor_sha256"] != digest(executor.read_bytes())
        for index, template in enumerate(templates)
    ):
        raise SystemExit("Executor or action identities differ from calibration.")

    source = repo / "D2R-RAG/src"
    current_source = {
        path.name: digest(path.read_bytes())
        for path in sorted(source.iterdir())
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    }
    if current_source != templates[0]["index"]["source"]:
        raise SystemExit("Upstream source differs from calibration.")
    for field, path in (
        ("corpus", prepared / "repair_corpus.jsonl"),
        ("secondary", prepared / "repair_secondary.jsonl"),
    ):
        if digest(path.read_bytes()) != templates[0]["index"][field]:
            raise SystemExit(f"The {field} pool differs from calibration.")

    drafts = {row["case_id"]: row for row in rows(draft_file)}
    prepared_cases = {
        row["case_id"]: row
        for row in rows(prepared / "draft_inputs.jsonl")
    }
    if any(
        cid not in prepared_cases
        or row["question"] != prepared_cases[cid]["question"]
        for cid, row in drafts.items()
    ):
        raise SystemExit("Prepared questions differ from the saved drafts.")

    audit = {
        row["case_id"]: row for row in rows(prepared / "audit.jsonl")
    }
    train_ids = set(split["train_question_ids"])
    validation_ids = set(split["validation_question_ids"])
    if train_ids & validation_ids:
        raise SystemExit("Training and validation question IDs overlap.")
    train_cases = [
        cid for cid, row in drafts.items()
        if audit[cid]["question_id"] in train_ids and repair_required(row)
    ]
    validation_cases = [
        cid for cid in drafts
        if audit[cid]["question_id"] in validation_ids
    ]
    if not train_cases:
        raise SystemExit("No flagged training cases.")

    budgets = {
        name: float(value) * args.budget_scale
        for name, value in calibration["base_budgets"].items()
    }
    if any(not np.isfinite(value) or value <= 0 for value in budgets.values()):
        raise SystemExit("Invalid calibrated budgets.")
    print("Budgets:", encode(budgets), flush=True)
    print("Flagged training cases:", len(train_cases),
          "| validation cases:", len(validation_cases), flush=True)
    if not args.run and not args.prepare_only:
        print("Preview complete. Use --prepare-only or --run.")
        return

    import torch
    from src.patcher import BanditPatcher
    from src.bandits import LinUCB

    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    if runtime != plan["inputs"]["runtime"]:
        raise SystemExit("Runtime differs from calibration. Use the calibrated environment.")

    config = {
        "drafts": digest(draft_file.read_bytes()),
        "split": digest(split_file.read_bytes()),
        "calibration": digest(calibration_file.read_bytes()),
        "templates": digest(encode(templates).encode()),
        "feature_code": digest(
            (repo / "reproduction/label_features.py").read_bytes()
        ),
        "trainer": digest(Path(__file__).read_bytes()),
        "budgets": budgets,
        "alpha": 2.0,
        "epochs": args.epochs,
        "seed": args.seed,
        "runtime": runtime,
    }

    output.mkdir(parents=True, exist_ok=True)
    with (output / "trainer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        feature_path = output / "features.json"
        if feature_path.exists():
            bank = json.loads(feature_path.read_text())
            if bank["config"] != config:
                raise SystemExit("Existing features use another configuration. Use a fresh output directory.")
        else:
            encoder = BanditPatcher(
                str(output),
                latency_budget=budgets["latency_s"],
                vram_budget=budgets["vram_mb"],
                method="linucb", alpha=2.0,
                with_gating=True, with_cost=True,
            )
            bank = {
                "config": config,
                "rows": {},
                "context_dim": encoder.context_dim,
            }
            for cid, draft in drafts.items():
                if not repair_required(draft):
                    continue
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                pair = paired_contexts(encoder, draft["question"], draft)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if pair["with_label"].size != encoder.context_dim:
                    raise RuntimeError("Context width differs from the upstream bandit width.")
                bank["rows"][cid] = {name: pair[name] for name in NAMES}
                bank["rows"][cid]["feature_seconds"] = time.perf_counter() - started
            save_json(feature_path, bank)
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Prepared features:", len(bank["rows"]),
              "cases x", bank["context_dim"], flush=True)
        if args.prepare_only:
            return

        ledger = repo / "results/smoke-budget.json"
        state_path = output / "journal.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["config"] != config:
                raise SystemExit("Training journal uses another configuration.")
        else:
            used = json.loads(ledger.read_text())["requests"]
            state = {
                "config": config,
                "request_cap": used + args.max_new_requests,
                "training": [],
                "validation": [],
            }
            save_json(state_path, state)

        print("Cumulative request cap:", state["request_cap"], flush=True)
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        database_path = shared / "rollouts.sqlite3"

        def context(cid, name):
            return np.asarray(bank["rows"][cid][name], dtype=float)

        def outcome(cid, action_id, kb_seed):
            identity = dict(templates[action_id])
            identity.update(
                case_id=cid,
                kb_seed=kb_seed,
                draft_sha256=digest(encode(drafts[cid]).encode()),
            )
            key = digest(encode(identity).encode())
            result = cached_result(database_path, key)
            if result is None:
                if json.loads(ledger.read_text())["requests"] >= state["request_cap"]:
                    raise RuntimeError("Training job reached its saved request cap.")
                log = logs / f"{cid}-seed{kb_seed}-action{action_id}.log"
                command = [
                    sys.executable, "-u", "-c", LAUNCHER,
                    str(prepared), str(draft_file), str(shared),
                    "--case-id", cid,
                    "--action-id", str(action_id),
                    "--with-kg", "--kb-seed", str(kb_seed),
                    "--max-new-requests", str(args.requests_per_action),
                    "--max-total-requests", str(state["request_cap"]),
                ]
                print(f"RUN {cid} action {action_id} seed {kb_seed}", flush=True)
                with log.open("a", encoding="utf-8") as stream:
                    completed = subprocess.run(
                        command, cwd=repo,
                        stdout=stream, stderr=subprocess.STDOUT,
                    )
                if completed.returncode:
                    print("\n".join(log.read_text().splitlines()[-25:]))
                    raise RuntimeError(f"Repair failed. Inspect {log} and rerun to resume.")
                result = cached_result(database_path, key)
                if result is None:
                    raise RuntimeError("Repair did not create the expected cache entry.")
            return key, result

        provider = SimpleNamespace(
            latency_budget=budgets["latency_s"],
            vram_budget=budgets["vram_mb"],
            with_gating=True,
            with_cost=True,
            method="linucb",
        )

        def reward(result):
            response = result["response"]
            label, _ = BanditPatcher.get_failure_label(None, response)
            return BanditPatcher.calculate_reward(
                provider, label, response["consistency_check"],
                response["entailment_check"]["response"],
                response["latency"], response["vram_usage"],
            )

        models = {
            name: LinUCB(11, bank["context_dim"], alpha=2.0)
            for name in NAMES
        }
        arms = list(range(11))
        schedule = []
        for epoch in range(args.epochs):
            order = train_cases.copy()
            random.Random(args.seed + epoch).shuffle(order)
            schedule.extend((epoch, cid) for cid in order)

        for index, step in enumerate(state["training"]):
            if (index >= len(schedule)
                    or (step["epoch"], step["case_id"]) != schedule[index]):
                raise RuntimeError("Training journal differs from the expected schedule.")
            for name in NAMES:
                models[name].update(
                    step["choices"][name],
                    context(step["case_id"], name),
                    step["rewards"][name]["total_reward"],
                )

        for index in range(len(state["training"]), len(schedule)):
            epoch, cid = schedule[index]
            choices = {
                name: models[name].select_arm(context(cid, name), arms)
                for name in NAMES
            }
            outcomes = {
                action: outcome(cid, action, args.seed + epoch)
                for action in set(choices.values())
            }
            rewards = {
                name: reward(outcomes[choices[name]][1])
                for name in NAMES
            }
            for name in NAMES:
                models[name].update(
                    choices[name], context(cid, name),
                    rewards[name]["total_reward"],
                )
            state["training"].append({
                "epoch": epoch,
                "case_id": cid,
                "choices": choices,
                "rewards": rewards,
                "outcome_keys": {
                    name: outcomes[choices[name]][0] for name in NAMES
                },
            })
            save_json(state_path, state)
            print(f"TRAIN {index + 1}/{len(schedule)} {choices}", flush=True)

        for index in range(len(state["validation"]), len(validation_cases)):
            cid = validation_cases[index]
            draft = drafts[cid]
            decisions = {}
            for name in NAMES:
                action, routing_seconds = None, 0.0
                if repair_required(draft):
                    started = time.perf_counter()
                    action = models[name].select_arm(context(cid, name), arms)
                    routing_seconds = time.perf_counter() - started
                    key, result = outcome(cid, action, args.seed)
                    answer = result["response"]["response"]
                    repair_seconds = result["wall_seconds"]
                    repair_input = result["input_tokens"]
                    repair_output = result["output_tokens"]
                    feature_seconds = bank["rows"][cid]["feature_seconds"]
                else:
                    key, answer = None, draft["response"]
                    repair_seconds = feature_seconds = 0.0
                    repair_input = repair_output = 0
                decisions[name] = {
                    "action_id": action,
                    "answer": answer,
                    "outcome_key": key,
                    "replayed_seconds": (
                        draft["end_to_end_latency_s"] + feature_seconds
                        + routing_seconds + repair_seconds
                    ),
                    "input_tokens": draft["input_tokens"] + repair_input,
                    "output_tokens": draft["output_tokens"] + repair_output,
                }
            state["validation"].append({
                "case_id": cid,
                "question_id": audit[cid]["question_id"],
                "condition": audit[cid]["condition"],
                "variants": decisions,
            })
            save_json(state_path, state)
            print(f"VALIDATION {index + 1}/{len(validation_cases)}", flush=True)

        for name, model in models.items():
            target = output / f"{name}.pkl"
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump({
                    "config": config, "variant": name, "model": model,
                }, stream)
            os.replace(temporary, target)

        target = output / "validation.jsonl"
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in state["validation"]:
                stream.write(encode(row) + "\n")
        os.replace(temporary, target)

        print("Training steps:", len(state["training"]))
        for name in NAMES:
            average = np.mean([
                row["variants"][name]["replayed_seconds"]
                for row in state["validation"]
            ])
            print(name, "| replayed mean seconds:", float(average))
        print("Saved:", target)


if __name__ == "__main__":
    main()
