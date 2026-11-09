"""Reuse RAGAS scores only for identical inputs and judge settings."""
import ast
from contextlib import closing
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "reproduction"))
import evaluate_saved_d2r as ev

old_run = Path.home() / "dsc180-runs/label-policy-pilot-01"
new_run = Path.home() / "dsc180-runs/label-policy-pilot-02"
old_folder = old_run / "offline-ragas-01"
folder = new_run / "offline-ragas-01"
old_protocol = json.loads(
    (old_folder / "plan.json").read_text()
)["protocol"]

source = new_run / "evaluation_inputs.jsonl"
mapping_path = new_run / "evaluation_mapping.jsonl"
exported = json.loads(
    (new_run / "evaluation_export.json").read_text()
)
versions = {
    name: version(name) for name in old_protocol["versions"]
}
if versions != exported["versions"]:
    raise RuntimeError("Installed versions differ from the export")
if ev.digest(source.read_bytes()) != exported["evaluation_inputs_sha256"]:
    raise RuntimeError("New exported inputs changed")
if ev.digest(
    (old_run / "evaluation_inputs.jsonl").read_bytes()
) != old_protocol["inputs_sha256"]:
    raise RuntimeError("Original exported inputs changed")
if ev.digest(
    (old_run / "evaluation_mapping.jsonl").read_bytes()
) != old_protocol["mapping_sha256"]:
    raise RuntimeError("Original mapping changed")

# Read the current protocol dictionary without running evaluation.
tree = ast.parse(Path(ev.__file__).read_text())
main = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "main"
)
node = next(
    n for n in main.body
    if isinstance(n, ast.Assign)
    and any(
        isinstance(t, ast.Name) and t.id == "protocol"
        for t in n.targets
    )
)
namespace = dict(
    vars(ev), root=root, source=source,
    mapping_path=mapping_path, versions=versions,
)
exec(
    compile(
        ast.Module(body=[node], type_ignores=[]),
        ev.__file__, "exec"
    ),
    namespace,
)
protocol = namespace["protocol"]
ignored = {"inputs_sha256", "mapping_sha256"}
if (
    {k: v for k, v in protocol.items() if k not in ignored}
    != {k: v for k, v in old_protocol.items() if k not in ignored}
):
    raise RuntimeError("Judge settings differ; cannot reuse scores")

def read_inputs(path):
    return {
        item["evaluation_id"]: item
        for item in (
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    }

old_items = read_inputs(old_run / "evaluation_inputs.jsonl")
new_items = read_inputs(source)
common = sorted(set(old_items) & set(new_items))
if any(old_items[key] != new_items[key] for key in common):
    raise RuntimeError("Matching IDs contain different data")

folder.mkdir(exist_ok=True)
plan_path = folder / "plan.json"
if (
    plan_path.exists()
    and json.loads(plan_path.read_text())["protocol"] != protocol
):
    raise RuntimeError("Target evaluation uses another protocol")

additions = []
with (folder / "evaluation.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with closing(sqlite3.connect(
        (old_folder / "scores.sqlite3").as_uri() + "?mode=ro",
        uri=True,
    )) as old:
        for eid in common:
            for metric in protocol["metrics"]:
                old_key = ev.digest(
                    ev.dump([old_protocol, eid, metric]).encode()
                )
                found = old.execute(
                    "SELECT payload FROM scores WHERE key=?",
                    (old_key,),
                ).fetchone()
                if found:
                    key = ev.digest(
                        ev.dump([protocol, eid, metric]).encode()
                    )
                    additions.append((key, found[0]))

    with closing(sqlite3.connect(
        folder / "scores.sqlite3"
    )) as database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS scores "
            "(key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        for key, payload in additions:
            found = database.execute(
                "SELECT payload FROM scores WHERE key=?", (key,)
            ).fetchone()
            if (
                found
                and json.loads(found[0])["score"]
                != json.loads(payload)["score"]
            ):
                raise RuntimeError("Conflicting existing score")
        database.executemany(
            "INSERT OR IGNORE INTO scores VALUES (?, ?)",
            additions,
        )
        database.commit()

    (folder / "score_reuse.json").write_text(json.dumps({
        "source": str(old_folder),
        "matching_inputs": len(common),
        "reusable_metric_results": len(additions),
        "old_protocol": old_protocol,
        "new_protocol": protocol,
    }, indent=2) + "\n")

print("Matching answer-and-evidence inputs:", len(common))
print("Reusable RAGAS metric results:", len(additions))
print("No API calls made.")
