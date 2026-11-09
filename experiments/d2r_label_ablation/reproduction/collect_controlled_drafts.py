"""Collect shared controlled drafts with KG checks and a resumable SQLite cache."""
import argparse
from collections import Counter
import fcntl
import json
from importlib.metadata import version
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

from controlled_repair import BASE_PARAMS, digest, encode


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export_drafts(database, keys, target):
    rows = []
    for key in keys:
        cached = database.execute(
            "SELECT result FROM drafts WHERE key = ?", (key,)
        ).fetchone()
        if cached:
            rows.append(cached[0])
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(row + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("shared_repair", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--initialize-graph", action="store_true",
        help="Allow building the graph for a new corpus under the request guard.",
    )
    parser.add_argument("--max-new-requests", type=int, default=0)
    parser.add_argument("--max-total-requests", type=int, default=320)
    args = parser.parse_args()
    prepared, shared, output = (
        path.resolve() for path in
        (args.prepared, args.shared_repair, args.output)
    )
    repo = Path(__file__).resolve().parents[1]
    upstream = repo / "D2R-RAG"
    cases = read_rows(prepared / "draft_inputs.jsonl")
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise SystemExit("Expected distinct prepared cases.")
    if any(len(case["passages"]) != 4 for case in cases):
        raise SystemExit("Expected four initial passages per case.")
    groups = Counter(case["question"] for case in cases)
    if any(count != 3 for count in groups.values()):
        raise SystemExit("Expected three conditions per question.")

    secondary = prepared / "repair_secondary.jsonl"
    signature = {
        "corpus": digest((prepared / "repair_corpus.jsonl").read_bytes()),
        "secondary": digest(secondary.read_bytes()) if secondary.exists() else None,
        "source": {
            path.name: digest(path.read_bytes())
            for path in sorted((upstream / "src").iterdir())
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
        },
    }
    index_key = digest(encode(signature).encode())[:16]
    graph_dir = shared / f"graph-{index_key}"
    if not graph_dir.exists() and not args.initialize_graph:
        raise SystemExit(
            f"Graph is missing: {graph_dir}. "
            "Use --initialize-graph for a new corpus."
        )

    import torch
    runtime = {
        "python": sys.version,
        "cpu_threads": 4,
        "packages": {
            name: version(name)
            for name in ("torch", "openai", "llama-index-core")
        },
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    executor_hash = digest(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("kg_setup_cache.py").read_bytes()
    )
    keys = [
        digest(encode({
            "case": case, "index": signature, "params": BASE_PARAMS,
            "kg_enabled": True, "executor": executor_hash, "runtime": runtime,
        }).encode())
        for case in cases
    ]
    if args.dry_run:
        print(f"Prepared: {len(cases)} cases across {len(groups)} questions.")
        print("KG directory:", graph_dir)
        print("Output:", output)
        return
    if args.max_new_requests <= 0 or args.max_total_requests <= 0:
        parser.error("Specify positive request limits for collection.")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(output / "drafts.sqlite3") as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS drafts "
                "(key TEXT PRIMARY KEY, result TEXT NOT NULL)"
            )
            database.commit()
            export_drafts(database, keys, output / "controlled_drafts.jsonl")
            rag = None

            for case, key in zip(cases, keys):
                if database.execute(
                    "SELECT 1 FROM drafts WHERE key = ?", (key,)
                ).fetchone():
                    print("CACHED", case["case_id"], flush=True)
                    continue

                if rag is None:
                    import budget_guard
                    ledger = repo / "results" / "smoke-budget.json"
                    used = json.loads(ledger.read_text())["requests"]
                    if used >= args.max_total_requests:
                        raise SystemExit("Configured cumulative request limit reached.")
                    attempt = Path(tempfile.mkdtemp(prefix="drafts-", dir=output))
                    trace = attempt / "llm_chatlog.jsonl"
                    budget_guard.MAX_REQUESTS = min(
                        args.max_total_requests, used + args.max_new_requests
                    )
                    budget_guard.install(
                        ledger, chat_log=trace, case_phase="draft_case"
                    )
                    sys.path.insert(0, str(upstream))
                    os.chdir(upstream)
                    from src.knowledge_graph import KnowledgeGraph
                    from src.rag_engine import RAGEngine
                    from src.utils import KnowledgeBaseSimulator, setup_settings

                    setup = setup_settings("hotpotqa")
                    graph_store = (
                        graph_dir / setup["kg_storage_dir"] / "index_store.json"
                    )
                    if not graph_store.is_file() and not args.initialize_graph:
                        raise SystemExit(f"Shared KG storage is incomplete: {graph_store}")
                    passages = read_rows(prepared / "repair_corpus.jsonl")
                    texts = [f"{row['title']}\n{row['text']}" for row in passages]
                    kb = KnowledgeBaseSimulator(texts, [], random_seed=180)
                    graph_dir.mkdir(parents=True, exist_ok=True)
                    graph = KnowledgeGraph(str(graph_dir), **setup)
                    from kg_setup_cache import build_graph_with_cache
                    build_graph_with_cache(graph, kb, graph_dir, signature)
                    index = shared / f"index-{index_key}"
                    index.mkdir(exist_ok=True)
                    rag = RAGEngine(
                        str(index), kb, knowledge_graph=graph, **setup
                    )

                token = budget_guard.chat_case.set(case["case_id"])
                try:
                    started = time.perf_counter()
                    draft = rag.query_shortanswer(
                        case["question"], params=BASE_PARAMS.copy(),
                        consistency_check=True, entailment_check=True,
                        evidence_override=case["passages"],
                    )
                    elapsed = time.perf_counter() - started
                    observed = [
                        node["node_id"] for node in draft["retrieved_context"]
                    ]
                    expected = [
                        passage["passage_id"] for passage in case["passages"]
                    ]
                    if observed != expected:
                        raise RuntimeError("Initial evidence order differs from the prepared case.")
                    if "consistency_check" not in draft or "entailment_check" not in draft:
                        raise RuntimeError("Required verifier outputs are missing.")
                    usage = [
                        row["response"]["usage"] for row in read_rows(trace)
                        if row.get("case_id") == case["case_id"]
                        and row.get("status_code") == 200
                        and isinstance(row.get("response"), dict)
                        and row["response"].get("usage")
                    ]
                    if not usage:
                        raise RuntimeError("No token usage was logged for this case.")
                    record = {
                        **draft,
                        "case_id": case["case_id"],
                        "question": case["question"],
                        "end_to_end_latency_s": elapsed,
                        "chat_log": str(trace),
                        "input_tokens": sum(row["prompt_tokens"] for row in usage),
                        "output_tokens": sum(row["completion_tokens"] for row in usage),
                        "kg_enabled": True,
                        "cache_key": key,
                    }
                    database.execute(
                        "INSERT INTO drafts VALUES (?, ?)", (key, encode(record))
                    )
                    database.commit()
                except Exception as error:
                    failure = {
                        "case_id": case["case_id"],
                        "error_type": type(error).__name__,
                    }
                    (attempt / f"{case['case_id']}.failure.json").write_text(
                        encode(failure)
                    )
                    raise
                finally:
                    budget_guard.chat_case.reset(token)

                export_drafts(database, keys, output / "controlled_drafts.jsonl")
                print("SAVED", case["case_id"], flush=True)

    print("Complete:", output / "controlled_drafts.jsonl")


if __name__ == "__main__":
    main()
