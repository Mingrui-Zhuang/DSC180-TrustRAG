"""Check prerequisites without downloading models or calling paid APIs."""
import argparse
import ast
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    failures = []
    repo = ROOT / "D2R-RAG"
    for path in repo.rglob("*.py"):
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as error:
            failures.append(f"Syntax error: {path.relative_to(ROOT)}: {error}")
    prompts = repo / "src/prompts_shortanswer.yaml"
    if not prompts.is_file():
        failures.append("Missing upstream src/prompts_shortanswer.yaml. Obtain the authors' prompts before a faithful HotpotQA reproduction.")
    elif args.environment:
        try:
            import yaml
            values = yaml.safe_load(prompts.read_text())
            for key in ("simple_qa", "paraphrase_qa", "simplify_qa"):
                if not isinstance(values, dict) or not isinstance(values.get(key), str):
                    failures.append(f"Prompt key missing or invalid: {key}")
        except Exception as error:
            failures.append(f"Cannot read prompts: {type(error).__name__}: {error}")
    if not (repo / "src/prompts.yaml").is_file():
        print("NOTE: FEVER prompts (src/prompts.yaml) are also absent; initial target is HotpotQA.")
    if args.environment:
        if not os.environ.get("OPENAI_API_KEY"):
            failures.append("OPENAI_API_KEY is not set (value is never printed).")
        # Import libraries only: do not initialize models or run upstream code.
        for module in ("torch", "datasets", "faiss", "yaml", "drqa.retriever.utils",
                       "llama_index.llms.openai", "llama_index.embeddings.huggingface",
                       "llama_index.vector_stores.faiss", "llama_index.retrievers.bm25",
                       "sentence_transformers", "pyvis", "json_repair", "Stemmer",
                       "cloudpickle", "qa_metrics.em", "ragas.llms.base",
                       "ragas.embeddings", "ragas.prompt.pydantic_prompt"):
            try:
                importlib.import_module(module)
            except Exception as error:
                failures.append(f"Import {module}: {type(error).__name__}: {error}")
        if args.require_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    failures.append("CUDA GPU unavailable inside this container.")
            except ImportError:
                pass
    print("Static source checks completed; this does not validate an experiment.")
    for failure in failures:
        print("BLOCKED:", failure)
    if failures:
        return 1
    print("Preflight passed. Run a small experiment before claiming reproduction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
