"""Run upstream baseline on two questions, isolated from reproduction outputs.

This deliberately reduced dataset is an integration check, not a paper result.
It still calls the paid API and may download full underlying dataset shards.
"""
from pathlib import Path
import argparse
import os
import sys

# Set before NumPy/PyTorch imports to avoid oversubscribing the DSMLP CPUs.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'


def main():
    from budget_guard import install
    repo = Path(__file__).resolve().parents[1] / "D2R-RAG"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--stage', choices=['baseline', 'train', 'repair'], default='baseline')
    parser.add_argument("--controlled-inputs",
                        type=lambda value: Path(value).expanduser().resolve())
    args = parser.parse_args()
    if args.controlled_inputs is not None and args.stage != "baseline":
        parser.error("--controlled-inputs requires --stage baseline")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.stage != 'baseline')
    chat_log = output / "llm_chatlog.jsonl" if args.controlled_inputs else None
    install(repo.parent / "results" / "smoke-budget.json", chat_log=chat_log)
    import datasets
    if not (output / 'src').exists():
        (output / "src").symlink_to(repo / "src", target_is_directory=True)
    (output / "files_hotpotqa_v").mkdir(exist_ok=True)
    (output / "files_hotpotqa_t").mkdir(exist_ok=True)
    original_loader = datasets.load_dataset

    def limited_loader(path, *args, **kwargs):
        if path != "hotpotqa/hotpot_qa":
            raise ValueError("Smoke runner only supports upstream HotpotQA")
        requested = kwargs["split"]
        split = requested.split("[", 1)[0]
        size = 5 if "100000" in requested else 2
        kwargs["split"] = f"{split}[:{size}]"
        return original_loader(path, *args, **kwargs)

    datasets.load_dataset = limited_loader
    os.chdir(output)
    if args.controlled_inputs is not None:
        run_controlled_drafts(args.controlled_inputs)
        return

    script = {'baseline': 'experiments/analysis_shortanswer.py',
              'train': 'src/train_shortanswer.py',
              'repair': 'experiments/analysis_shortanswer_patched.py'}[args.stage]
    # The training script otherwise references params_updates before assignment
    # if the first example is diagnosed NO_FAILURE. Apply this control-flow fix
    # to the in-memory smoke copy only; retain the authors' source unchanged.
    source = (repo / script).read_text()
    if args.stage == 'train':
        target = '        for row in tqdm.tqdm(dataset):\n'
        if source.count(target) != 1:
            raise RuntimeError('Training source changed; review smoke compatibility fix')
        source = source.replace(target, target + '            params_updates = {}\n')
    sys.argv = [str(repo / script), "hotpotqa"]
    print(f"SMOKE ONLY ({args.stage}): 2 questions per split; secondary contexts from 5 examples.")
    exec(compile(source, sys.argv[0], 'exec'), {'__name__': '__main__', '__file__': sys.argv[0]})



def run_controlled_drafts(input_path):
    import json
    import time
    from itertools import islice

    from budget_guard import chat_case

    with input_path.open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in islice(stream, 3)]
    if (len(cases) != 3 or len({c["question"] for c in cases}) != 1
            or len({c["case_id"] for c in cases}) != 3
            or any(len(c["passages"]) != 4 for c in cases)):
        raise ValueError("Expected three distinct cases for one question, with four passages each.")

    from src.knowledge_graph import KnowledgeGraph
    from src.rag_engine import RAGEngine
    from src.utils import load_hotpotqa, setup_settings

    _, knowledgebase = load_hotpotqa(split="validation")
    setup = setup_settings("hotpotqa")
    folder = "files_hotpotqa_v"
    graph = KnowledgeGraph(folder, **setup)
    graph.build(knowledgebase)
    rag = RAGEngine(folder, knowledgebase, knowledge_graph=graph, **setup)
    params = {"retriever": "dense", "topk": 5, "reranker": False,
              "prompt_edit": "simple_qa", "reindex": False}

    with Path("controlled_drafts.jsonl").open("x", encoding="utf-8") as output:
        for case in cases:
            passages = case["passages"]
            started = time.perf_counter()
            token = chat_case.set(case["case_id"])
            try:
                draft = rag.query_shortanswer(
                    case["question"], params=params, consistency_check=True,
                    entailment_check=True, evidence_override=passages,
                )
            finally:
                chat_case.reset(token)
            elapsed = time.perf_counter() - started
            observed = [node["node_id"] for node in draft["retrieved_context"]]
            expected = [passage["passage_id"] for passage in passages]
            if observed != expected:
                raise RuntimeError(f"Source mismatch for {case['case_id']}: {observed}")
            output.write(json.dumps({
                "case_id": case["case_id"], "question": case["question"],
                "end_to_end_latency_s": elapsed, **draft,
            }, ensure_ascii=False) + "\n")
            output.flush()
            print(f"PASS {case['case_id']}: four expected source IDs", flush=True)


if __name__ == "__main__":
    main()
