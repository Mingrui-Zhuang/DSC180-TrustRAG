"""Serial, cached offline RAGAS evaluation of exported smoke outputs."""
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"


def dump(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(dump(value) + "\n")
    os.replace(temporary, path)


def report(folder, mapping, cached):
    scored = []
    for row in mapping:
        item = dict(row)
        for metric in ("answer_relevancy", "faithfulness"):
            item[metric] = cached[(row["evaluation_id"], metric)]["score"]
        scored.append(item)

    (folder / "scored_comparison.jsonl").write_text(
        "".join(dump(row) + "\n" for row in scored)
    )
    groups = {"overall": scored}
    for condition in sorted({row["condition"] for row in scored}):
        groups[condition] = [
            row for row in scored if row["condition"] == condition
        ]

    summary = {}
    for group, members in groups.items():
        summary[group] = {}
        print("\n" + group)
        for variant in ("baseline", "with_label", "without_label"):
            selected = [
                row for row in members if row["variant"] == variant
            ]
            result = {"n": len(selected)}
            for metric in ("answer_relevancy", "faithfulness"):
                values = [
                    row[metric]
                    for row in selected
                    if row[metric] is not None
                ]
                result[metric] = {
                    "mean": sum(values) / len(values) if values else None,
                    "defined": len(values),
                    "total": len(selected),
                }
            summary[group][variant] = result
            print(variant, dump(result))

    write_json(folder / "summary.json", summary)
    print("\nSaved:", folder / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy_run",
        type=lambda value: Path(value).expanduser().resolve(),
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=80)
    args = parser.parse_args()
    if args.max_new_requests <= 0:
        parser.error("--max-new-requests must be positive")

    root = Path(__file__).resolve().parents[1]
    run = args.policy_run
    source = run / "evaluation_inputs.jsonl"
    mapping_path = run / "evaluation_mapping.jsonl"
    exported = json.loads((run / "evaluation_export.json").read_text())

    if digest(source.read_bytes()) != exported["evaluation_inputs_sha256"]:
        raise RuntimeError("Exported evaluation inputs changed")

    inputs, mapping = rows(source), rows(mapping_path)
    known = {row["evaluation_id"] for row in inputs}
    if len(known) != len(inputs):
        raise RuntimeError("Duplicate exported evaluation IDs")
    if any(row["evaluation_id"] not in known for row in mapping):
        raise RuntimeError("Comparison references an unknown input")

    packages = (
        "ragas", "llama-index-core", "llama-index-llms-openai",
        "openai", "langchain-core",
    )
    versions = {name: metadata.version(name) for name in packages}
    if versions != exported["versions"] or versions["ragas"] != "0.2.15":
        raise RuntimeError("Package versions differ from the export")

    protocol = {
        "version": 1,
        "inputs_sha256": digest(source.read_bytes()),
        "mapping_sha256": digest(mapping_path.read_bytes()),
        "versions": versions,
        "evaluation_source_sha256": digest(
            (root / "D2R-RAG/src/evaluation.py").read_bytes()
        ),
        "model": "gpt-4o-mini",
        "max_output_tokens": 1024,
        "max_completions": 3,
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "metrics": ["answer_relevancy", "faithfulness"],
        "serial": True,
    }

    folder = run / "offline-ragas-01"
    folder.mkdir(exist_ok=True)
    with (folder / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(folder / "scores.sqlite3") as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS scores "
                "(key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            tasks, cached = [], {}
            for item in inputs:
                for metric in protocol["metrics"]:
                    key = digest(dump(
                        [protocol, item["evaluation_id"], metric]
                    ).encode())
                    found = database.execute(
                        "SELECT payload FROM scores WHERE key=?", (key,)
                    ).fetchone()
                    if found:
                        cached[(item["evaluation_id"], metric)] = json.loads(
                            found[0]
                        )
                    else:
                        tasks.append((key, item, metric))

            print("Comparison rows:", len(mapping),
                  "| unique inputs:", len(inputs))
            print("Cached metric results:", len(cached),
                  "| pending:", len(tasks))

            if not tasks:
                report(folder, mapping, cached)
                return
            if not args.run:
                print("Preview only. Add --run to evaluate saved outputs.")
                return
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")

            import budget_guard
            ledger = root / "results/smoke-budget.json"
            if not ledger.is_file():
                raise RuntimeError("Existing shared budget ledger is missing")
            state = json.loads(ledger.read_text())
            plan_path = folder / "plan.json"

            if plan_path.exists():
                plan = json.loads(plan_path.read_text())
                if plan["protocol"] != protocol:
                    raise RuntimeError("Evaluation protocol changed")
            else:
                plan = {
                    "protocol": protocol,
                    "request_cap": state["requests"] + args.max_new_requests,
                    "starting_ledger": state,
                    "created_unix": time.time(),
                }
                write_json(plan_path, plan)

            budget_guard.MAX_REQUESTS = plan["request_cap"]
            budget_guard.MAX_OUTPUT = 1024
            budget_guard.MAX_COMPLETIONS = 3
            trace = folder / (
                "llm_chatlog-" + str(time.time_ns()) + ".jsonl"
            )
            budget_guard.install(
                ledger, chat_log=trace, case_phase="offline_evaluation"
            )
            print("Cumulative request cap:", plan["request_cap"])
            print("Using the existing shared $1 reservation allowance.")

            sys.path.insert(0, str(root / "D2R-RAG"))
            import src.evaluation as upstream
            from ragas.run_config import RunConfig

            # Importing upstream installs the authors' RAGAS parser patch.
            llm = upstream.LlamaIndexLLMWrapper(
                upstream.OpenAI(
                    model="gpt-4o-mini", max_tokens=1024, max_retries=0
                )
            )
            device = (
                "cuda" if upstream.torch.cuda.is_available() else "cpu"
            )
            embeddings = upstream.LlamaIndexEmbeddingsWrapper(
                upstream.HuggingFaceEmbedding(
                    model_name=protocol["embedding"], device=device
                )
            )

            for key, item, metric in tasks:
                tag = item["evaluation_id"][:12] + ":" + metric
                print("RUN", tag, flush=True)
                token = budget_guard.chat_case.set(tag)
                started = time.perf_counter()
                try:
                    dataset = upstream.Dataset.from_list([{
                        name: item[name]
                        for name in (
                            "question", "answer", "contexts", "ground_truth"
                        )
                    }])
                    result = upstream.evaluate(
                        dataset=dataset,
                        metrics=[copy.deepcopy(getattr(upstream, metric))],
                        llm=llm,
                        embeddings=embeddings,
                        run_config=RunConfig(
                            max_workers=1, max_retries=1, timeout=180
                        ),
                        batch_size=1,
                        raise_exceptions=True,
                        show_progress=False,
                    )
                    value = float(result.to_pandas().iloc[0][metric])
                    score = value if math.isfinite(value) else None
                    payload = {
                        "evaluation_id": item["evaluation_id"],
                        "metric": metric,
                        "score": score,
                        "status": (
                            "defined" if score is not None else "undefined"
                        ),
                        "wall_seconds": time.perf_counter() - started,
                        "chat_log": str(trace),
                    }
                    database.execute(
                        "INSERT INTO scores VALUES (?, ?)",
                        (key, dump(payload)),
                    )
                    database.commit()
                    cached[(item["evaluation_id"], metric)] = payload
                    print("SAVED", tag, score, flush=True)

                except Exception as error:
                    write_json(folder / "last_failure.json", {
                        "task": tag,
                        "error_type": type(error).__name__,
                        "chat_log": str(trace),
                    })
                    print(
                        "Stopped; completed scores are cached. Chat log:",
                        trace, flush=True,
                    )
                    seen = set()
                    while error is not None and id(error) not in seen:
                        seen.add(id(error))
                        print(type(error).__name__, str(error), flush=True)
                        error = error.__cause__ or error.__context__
                    raise SystemExit(1)
                finally:
                    budget_guard.chat_case.reset(token)

            report(folder, mapping, cached)


if __name__ == "__main__":
    main()
