# D2R label-ablation implementation checkpoint

Compare LinUCB repair policies with and without an explicit failure label. Both retain the same KG/NLI diagnostic signals, repair actions, shared trigger, and reward definition.

## Restore upstream code

Run from this directory:

```bash
git clone https://github.com/CyberScienceLab/D2R-RAG.git D2R-RAG
git -C D2R-RAG checkout --detach 5573ce938cee8b79f2e73ccc05f8fff803f464ed
git -C D2R-RAG apply --check ../upstream.patch
git -C D2R-RAG apply ../upstream.patch
```

The exporter verified exact reconstruction of the current runtime source, including final-newline differences. See source_manifest.json.

## Environment

The pilot used Python 3.11, PyTorch 2.5.1 with CUDA 12.1, a GTX 1080 Ti, and GPT-4o-mini through the OpenAI API. requirements-reproduction.txt records project requirements. environment-installed.txt is an inventory, not a portable lockfile.

## Scope

This checkpoint contains source, patches, and environment metadata. Frozen inputs and historical drafts/calibration are included under inputs/. The original input-selection recipe, spending-ledger setup, and portable end-to-end training instructions remain unfinished. No API key, model cache, repair database, or training checkpoint is included.

Results: ../../results/label-ablation-three-seed-pilot-02/.


## Frozen pilot and results

The pilot used 32 training questions and 16 held-out validation questions,
with three initial evidence conditions: clean_complete, missing_support,
and complete_distractors. There were 49 flagged training cases and two
epochs, giving 98 updates per policy. Seeds were 180, 181, and 182 at
budget scale 1.0. Every seed used the same 48 validation cases.

inputs/prepared/ contains the controlled evidence, split, audit, corpus,
expansion pool, and preparation manifest. inputs/drafts/ contains the
historical drafts, frozen split, calibration plan, and action budgets.
inputs/manifest.json records file hashes. Embedded log paths refer to
the original DSMLP workspace.

Mean automated accuracy: baseline 41.67%, with label 35.42%, without label
39.58%. The label-minus-no-label difference was -4.17 percentage points,
with a question-cluster bootstrap interval of [-13.19, 2.78].

After correcting 18 verified grading errors, the corresponding accuracies
were 45.83%, 43.75%, and 46.53%; difference -2.78 points, interval
[-11.81, 3.47]. Remaining judgments have not all been manually verified.
Intervals condition on the three fitted seeds. No equivalence is established.

Recorded mean replayed latency gaps were below 5% within each seed.
Live timing verification remains unfinished. Repair may replace the
initial controlled evidence through ordinary retrieval.

Historical repair databases, persisted graphs, training checkpoints, and
complete chat logs remain in the original DSMLP workspace. This package
alone cannot replay all historical training outcomes. Multiple budgets
and a larger untouched test set remain future work.
