"""Prepare a corrected pilot using frozen drafts and a compatible saved KG."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory

root = Path.cwd()
home = Path.home()
prepared = home / "dsc180-runs/label-pilot-inputs-01"
old_drafts = home / "dsc180-runs/label-pilot-drafts-01"
old_shared = home / "dsc180-runs/label-pilot-shared-01"
old_policy = home / "dsc180-runs/label-policy-pilot-01"
new_drafts = home / "dsc180-runs/label-pilot-drafts-02"
new_shared = home / "dsc180-runs/label-pilot-shared-02"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def index_key(signature):
    encoded = json.dumps(
        signature, ensure_ascii=False, sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]

plan = json.loads((old_drafts / "calibration_plan.json").read_text())
old_signature = plan["inputs"]["identities"][0]["index"]
source = root / "D2R-RAG/src"
current = {
    p.name: sha(p)
    for p in sorted(source.iterdir())
    if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}
}
changed = {
    name for name in set(current) | set(old_signature["source"])
    if current.get(name) != old_signature["source"].get(name)
}
if changed != {"rag_engine.py"}:
    raise RuntimeError(
        f"Expected only the BM25 correction; found: {sorted(changed)}"
    )

original = (
    old_policy
    / "source_before_bm25_fix/D2R-RAG/src/rag_engine.py"
)
if sha(original) != old_signature["source"]["rag_engine.py"]:
    raise RuntimeError("Original source snapshot differs from the pilot")

marker = b"        retriever.similarity_top_k = similarity_top_k\n"
updated = (source / "rag_engine.py").read_bytes()
if (
    updated.count(marker) != 1
    or updated.replace(marker, b"", 1) != original.read_bytes()
):
    raise RuntimeError(
        "Source change exceeds the one-line BM25 correction"
    )

for field, name in (
    ("corpus", "repair_corpus.jsonl"),
    ("secondary", "repair_secondary.jsonl"),
):
    if sha(prepared / name) != old_signature[field]:
        raise RuntimeError(f"The {field} pool changed")

split = json.loads((old_drafts / "pilot_split.json").read_text())
if sha(old_drafts / "controlled_drafts.jsonl") != split["drafts_sha256"]:
    raise RuntimeError("Frozen drafts changed")

new_signature = dict(old_signature, source=current)
old_graph = old_shared / ("graph-" + index_key(old_signature))
new_graph = new_shared / ("graph-" + index_key(new_signature))
if (
    not old_graph.is_dir()
    or not list(old_graph.rglob("index_store.json"))
):
    raise RuntimeError(f"Completed original KG is missing: {old_graph}")

manifest = {
    "change": "BM25 cached top-k correction",
    "parent_policy_run": str(old_policy),
    "initial_drafts_sha256": split["drafts_sha256"],
    "split_sha256": sha(old_drafts / "pilot_split.json"),
    "old_index_signature": old_signature,
    "new_index_signature": new_signature,
    "reused_kg_source": str(old_graph),
    "reused_kg_target": str(new_graph),
    "kg_reuse_reason":
        "Corpus and KG code/configuration unchanged; "
        "only the BM25 top-k assignment changed.",
}
manifest_path = new_drafts / "rerun_manifest.json"
if (
    manifest_path.exists()
    and json.loads(manifest_path.read_text()) != manifest
):
    raise RuntimeError("Existing rerun manifest differs")

for name in ("controlled_drafts.jsonl", "pilot_split.json"):
    target = new_drafts / name
    if target.exists() and sha(target) != sha(old_drafts / name):
        raise RuntimeError(f"Existing rerun input differs: {target}")

new_drafts.mkdir(parents=True, exist_ok=True)
new_shared.mkdir(parents=True, exist_ok=True)
for name in ("controlled_drafts.jsonl", "pilot_split.json"):
    shutil.copy2(old_drafts / name, new_drafts / name)

if not new_graph.exists():
    with TemporaryDirectory(
        prefix=".kg-copy-", dir=new_shared
    ) as directory:
        staged = Path(directory) / "graph"
        shutil.copytree(
            old_graph, staged,
            ignore=shutil.ignore_patterns(
                "setup.lock", "setup_triplets.sqlite3*"
            ),
        )
        staged.rename(new_graph)

for path in old_graph.rglob("*"):
    if (
        path.is_file()
        and path.name != "setup.lock"
        and not path.name.startswith("setup_triplets.sqlite3")
    ):
        target = new_graph / path.relative_to(old_graph)
        if not target.is_file() or sha(target) != sha(path):
            raise RuntimeError(f"Reused KG file differs: {target}")

with sqlite3.connect(new_shared / "rollouts.sqlite3") as database:
    database.execute(
        "CREATE TABLE IF NOT EXISTS repairs "
        "(key TEXT PRIMARY KEY, result TEXT NOT NULL)"
    )
    database.commit()
    cached = database.execute(
        "SELECT COUNT(*) FROM repairs"
    ).fetchone()[0]

manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("Reused frozen drafts:", new_drafts)
print("Shared KG copied and verified:", new_graph)
print("Corrected repair outcomes already cached:", cached)
print("Original pilot preserved:", old_policy)

ledger = json.loads((root / "results/smoke-budget.json").read_text())
print("Requests used:", ledger["requests"])
print("Reserved USD:", ledger["reserved_nanodollars"] / 1e9)
print(
    "Remaining reservation USD:",
    (ledger["limit_nanodollars"]
     - ledger["reserved_nanodollars"]) / 1e9
)
print("Preparation complete. No API calls made.")
