# DSC180 Trustworthy RAG

UC San Diego DSC180A capstone project on diagnosing and correcting failures in retrieval-augmented generation (RAG).

## Project Google Doc

[Project document](https://docs.google.com/document/d/1ohdW_3jHLRIEzxRnfOUgj9z-SVmoA4Xx6PHHZ1OWyjs/edit?tab=t.7dyc803eynwn)

We plan to reproduce a selected Doctor-RAG experiment, then run a small controlled experiment comparing explicit failure diagnosis with direct correction selection.

## Doctor-RAG reproduction

Doctor-RAG diagnoses a failed agentic RAG trajectory, predicts the error type and earliest failure point, and applies a targeted repair while reusing earlier valid work.

- Paper: [Doctor-RAG](https://arxiv.org/abs/2604.00865v2)
- Authors' repository: [dr_rag_open](https://github.com/Fdioa/dr_rag_open)

| Component | Proposed choice |
| --- | --- |
| Backbone | Qwen3-8B |
| Initial agent | ReAct |
| Dataset | HotpotQA |
| Comparison | Full rerun versus Doctor-RAG local repair |
| Evaluation | Answer exact match, token F1, repair rate, and inference time |

## Minimal controlled experiment

### Research question

Under which evidence conditions does diagnosing a failure before correction outperform selecting a correction directly, using the same models, actions, and computation budget?
