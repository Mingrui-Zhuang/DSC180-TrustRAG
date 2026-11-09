"""Checkpoint triplet extraction during shared graph setup only."""
import fcntl
import json
from pathlib import Path
import sqlite3
from controlled_repair import digest, encode


def build_graph_with_cache(graph, knowledgebase, graph_dir, signature):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    cache_path = graph_dir / "setup_triplets.sqlite3"

    with (graph_dir / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        database = sqlite3.connect(cache_path)
        database.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(key TEXT PRIMARY KEY, triplets TEXT NOT NULL)"
        )
        database.commit()
        original = graph.triplet_extractor.extract_triplets
        active = [True]
        counts = {"cached_chunks": 0, "new_chunks": 0}

        def extract(text, *args, **kwargs):
            if not active[0]:
                return original(text, *args, **kwargs)

            key = digest(encode(
                [1, signature, text, args, kwargs]
            ).encode())
            found = database.execute(
                "SELECT triplets FROM chunks WHERE key=?", (key,)
            ).fetchone()

            if found:
                counts["cached_chunks"] += 1
                return [tuple(item) for item in json.loads(found[0])]

            triplets = list(original(text, *args, **kwargs))
            if any(
                len(item) != 3
                or not all(isinstance(x, str) for x in item)
                for item in triplets
            ):
                raise RuntimeError("Unexpected extracted triplet format")

            database.execute(
                "INSERT INTO chunks VALUES (?, ?)",
                (key, encode(triplets)),
            )
            database.commit()
            counts["new_chunks"] += 1
            return triplets

        graph.triplet_extractor.extract_triplets = extract
        try:
            return graph.build(knowledgebase)
        finally:
            active[0] = False
            graph.triplet_extractor.extract_triplets = original
            database.close()
            print("KG setup extraction:", encode(counts), flush=True)
            print("KG setup cache:", cache_path, flush=True)
