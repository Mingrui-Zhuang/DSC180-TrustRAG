import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time


BASE_PARAMS = {
    "retriever": "dense", "topk": 5, "reindex": False,
    "reranker": False, "prompt_edit": "simple_qa",
}
ACTION_UPDATES = [
    {"retriever": retriever, "topk": topk, "reindex": reindex}
    for retriever in ("dense", "bm25")
    for topk in (10, 20)
    for reindex in (False, True)
] + [
    {"reranker": True},
    {"prompt_edit": "paraphrase_qa"},
    {"prompt_edit": "simplify_qa"},
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_default(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=_json_default
    )


def read_case(path, case_id):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["case_id"] == case_id:
                return row
    raise SystemExit(f"Case {case_id} was not found in {path}")


def run_repair(repo, prepared, output, case, index_key, params, args):
    import budget_guard

    ledger = repo / "results" / "smoke-budget.json"
    used = json.loads(ledger.read_text())["requests"]
    if used >= args.max_total_requests:
        raise SystemExit("The smoke ledger has reached the configured request limit.")

    attempt = Path(tempfile.mkdtemp(prefix="repair-", dir=output))
    trace = attempt / "llm_chatlog.jsonl"
    print(f"Repair attempt directory: {attempt}", flush=True)
    budget_guard.MAX_REQUESTS = min(args.max_total_requests, used + args.max_new_requests)
    budget_guard.install(ledger, chat_log=trace, case_phase="repair_case")

    source = repo / "D2R-RAG"
    sys.path.insert(0, str(source))
    os.chdir(source)
    from src.rag_engine import RAGEngine
    from src.utils import KnowledgeBaseSimulator, setup_settings

    with (prepared / "repair_corpus.jsonl").open(encoding="utf-8") as stream:
        passages = [json.loads(line) for line in stream]
    texts = [f"{row['title']}\n{row['text']}" for row in passages]
    secondary_path = prepared / "repair_secondary.jsonl"
    secondary = []
    if secondary_path.exists():
        with secondary_path.open(encoding="utf-8") as stream:
            secondary = [json.loads(line) for line in stream]
    secondary_texts = [f"{row['title']}\n{row['text']}" for row in secondary]
    kb = KnowledgeBaseSimulator(texts, secondary_texts, random_seed=args.kb_seed)
    graph = None
    if args.with_kg:
        from src.knowledge_graph import KnowledgeGraph
        graph_dir = output / f"graph-{index_key}"
        graph_dir.mkdir(exist_ok=True)
        graph = KnowledgeGraph(str(graph_dir), **setup_settings("hotpotqa"))
        graph.build(kb)
    index = output / f"index-{index_key}"
    index.mkdir(exist_ok=True)
    rag = RAGEngine(str(index), kb, knowledge_graph=graph, **setup_settings("hotpotqa"))

    token = budget_guard.chat_case.set(case["case_id"])
    kb.set_random_seed()
    started = time.perf_counter()
    try:
        repaired = rag.query_shortanswer(
            case["question"], params=params.copy(),
            consistency_check=args.with_kg, entailment_check=True,
        )
    finally:
        budget_guard.chat_case.reset(token)
    elapsed = time.perf_counter() - started

    with trace.open(encoding="utf-8") as stream:
        calls = [json.loads(line) for line in stream]
    usage = [
        row["response"]["usage"] for row in calls
        if row.get("phase") == "repair_case"
        and row.get("case_id") == case["case_id"]
        and row.get("status_code") == 200
        and isinstance(row.get("response"), dict)
        and row["response"].get("usage")
    ]
    if not usage:
        raise RuntimeError(f"No repair token usage in {trace}")

    return {
        "response": repaired,
        "kg_enabled": args.with_kg,
        "kb_before_size": len(texts),
        "kb_after_size": len(kb.get_current_kb()),
        "kb_changed": kb.get_current_kb() != texts,
        "kb_after_sha256": digest(encode(kb.get_current_kb()).encode()),
        "wall_seconds": elapsed,
        "input_tokens": sum(row["prompt_tokens"] for row in usage),
        "output_tokens": sum(row["completion_tokens"] for row in usage),
        "chat_log": str(trace),
    }


def main():
    parser = argparse.ArgumentParser(description="Run or reuse one upstream D2R-RAG repair action.")
    parser.add_argument("prepared", type=Path)
    parser.add_argument("drafts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--action-id", type=int, choices=range(11), default=0)
    parser.add_argument("--with-kg", action="store_true")
    parser.add_argument("--kb-seed", type=int, default=180)
    parser.add_argument("--max-new-requests", type=int, default=3)
    parser.add_argument("--max-total-requests", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_new_requests <= 0 or args.max_total_requests <= 0:
        parser.error("Request limits must be positive.")

    repo = Path(__file__).resolve().parents[1]
    prepared, drafts, output = (path.resolve() for path in (args.prepared, args.drafts, args.output))
    case = read_case(prepared / "draft_inputs.jsonl", args.case_id)
    draft = read_case(drafts, args.case_id)
    params = dict(BASE_PARAMS, **ACTION_UPDATES[args.action_id])
    secondary_path = prepared / "repair_secondary.jsonl"
    if params["reindex"] and not secondary_path.is_file():
        raise SystemExit("Build repair_secondary.jsonl before using a reindex action.")
    source = repo / "D2R-RAG" / "src"
    index_inputs = {
        "corpus": digest((prepared / "repair_corpus.jsonl").read_bytes()),
        "secondary": digest(secondary_path.read_bytes()) if secondary_path.exists() else None,
        "source": {
            path.name: digest(path.read_bytes())
            for path in sorted(source.iterdir())
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
        },
    }
    index_key = digest(encode(index_inputs).encode())[:16]
    identity = {
        "case_id": args.case_id,
        "action_id": args.action_id,
        "kg_enabled": args.with_kg,
        "kb_seed": args.kb_seed,
        "params": params,
        "draft_sha256": digest(encode(draft).encode()),
        "index": index_inputs,
        "executor_sha256": digest(Path(__file__).read_bytes()),
    }
    key = digest(encode(identity).encode())
    if args.dry_run:
        print(encode({
            "case_id": args.case_id, "action_id": args.action_id,
            "params": params, "kg_enabled": args.with_kg,
            "kb_seed": args.kb_seed, "secondary_present": secondary_path.exists(),
            "max_new_requests": args.max_new_requests,
            "max_total_requests": args.max_total_requests,
        }))
        return
    output.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(output / "rollouts.sqlite3", timeout=60) as cache:
        cache.execute("CREATE TABLE IF NOT EXISTS repairs (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
        cache.execute("BEGIN IMMEDIATE")
        existing = cache.execute("SELECT result FROM repairs WHERE key = ?", (key,)).fetchone()
        if existing:
            result = json.loads(existing[0])
            print("Cache hit.")
        else:
            result = run_repair(repo, prepared, output, case, index_key, params, args)
            result["identity"] = identity
            cache.execute("INSERT INTO repairs VALUES (?, ?)", (key, encode(result)))

    print("Draft:", draft["response"])
    print("Repair:", result["response"]["response"])
    print("Tokens:", result["input_tokens"], "input /", result["output_tokens"], "output")
    print("Chat log:", result["chat_log"])


if __name__ == "__main__":
    main()
