# Pilot closeout

Status: completed within the single-budget, three-seed pilot scope.

The experiment used 32 training questions and the same 16 validation
questions under three evidence conditions, with seeds 180, 181, and 182.
Both policies retained the same underlying diagnostics and shared gate.

The automated label-minus-no-label accuracy difference was -4.17 percentage
points, with a question-bootstrap 95% interval of [-13.19, 2.78].
After 18 verified judgment corrections, the difference was -2.78 points,
with interval [-11.81, 3.47]. No reliable label advantage was established.
This does not establish equivalence or show that labels are generally useless.

The seed-181 sample review agreed with the grader on 19/20 inputs;
seed 182 agreed on 18/20. The remaining full audit was deferred.
These results are partially audited. Original scores and the separate
sensitivity analysis are preserved.

Intervals condition on the three fitted seeds and resample the 16 questions.
Mean replayed latency gaps were below 5% within each seed. Live timing and
condition-specific cost matching were not completed. Evidence construction
used word-count matching and lexical-overlap ranking.

Official HotpotQA EM/F1 reporting, complete evidence auditing, portable
end-to-end setup, multiple budgets, and the larger untouched test study
remain outside this completed pilot scope.

Source reconstruction, frozen-input hashes, scoring-result hashes, and
the exported BM25 regression check passed before publication.
The raw checkpoint remains on DSMLP as label-pilot-raw-checkpoint-01.tar.gz.
Its checksum is recorded in RAW_ARCHIVE.sha256.
