# Controlled label-ablation pilot

Three fitted seeds (180–182), each trained on 32 questions and evaluated on the same 16 held-out questions under three evidence conditions. There are 48 validation cases per seed.

The mean label-minus-no-label accuracy difference is -4.17 percentage points under automated grading and -2.78 points after correcting 18 verified erroneous judgments. The corresponding question-cluster bootstrap intervals are [-13.19, 2.78] and [-11.81, 3.47].

The sensitivity analysis corrects known errors only. Remaining judgments have not all been manually verified. Bootstrap intervals condition on the three fitted seeds. Mean replayed latency gaps are below 5% within each seed; live timing verification remains unfinished.

This pilot tests adding an explicit failure label to shared diagnostic inputs. It does not establish equivalence or complete the larger study. RAGAS results, implementation code, and training artifacts are outside this compact scoring export.
