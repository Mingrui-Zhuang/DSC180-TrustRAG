import ast
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import Stemmer
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

root = Path.cwd()
old = (
    Path.home()
    / "dsc180-runs/label-policy-pilot-01"
    / "source_before_bm25_fix/D2R-RAG/src/rag_engine.py"
)
new = root / "D2R-RAG/src/rag_engine.py"

def exercise(path):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RAGEngine"
    )
    method = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "build_bm25_retriever"
    )
    namespace = {
        "os": os, "Stemmer": Stemmer,
        "BM25Retriever": BM25Retriever,
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path), "exec"
        ),
        namespace,
    )
    build = namespace["build_bm25_retriever"]

    with TemporaryDirectory() as directory:
        instance = SimpleNamespace(
            output_path=directory,
            kwargs={"bm25_retriever_storage": "bm25"},
            first_time_bm25_index=True,
            all_nodes=[
                TextNode(
                    text=f"Nebula telescope astronomy archive specimen {i}",
                    id_=str(i),
                )
                for i in range(32)
            ],
        )
        build(instance, 10)
        result = []
        for requested in (5, 10, 20):
            retriever = build(instance, requested)
            actual = len(retriever.retrieve("nebula telescope"))
            result.append(
                (requested, retriever.similarity_top_k, actual)
            )
        return result

with patch(
    "socket.socket.connect",
    side_effect=RuntimeError("Network disabled during BM25 test"),
):
    before = exercise(old)
    after = exercise(new)

print("Requested, configured, returned:")
print("Original:", before)
print("Fixed:   ", after)

assert any(
    requested != configured or requested != returned
    for requested, configured, returned in before
), "Original mismatch was not reproduced"
assert after == [(5, 5, 5), (10, 10, 10), (20, 20, 20)]

print("PASS: cached BM25 honors k=5, 10, and 20; no LLM or network calls.")
