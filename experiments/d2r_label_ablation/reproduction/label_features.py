"""Shared routing signals and paired LinUCB inputs for the label ablation."""
import numpy as np


def _signals(response):
    entailment = response.get("entailment_check", {})
    consistency = response.get("consistency_check", "MISSING")
    query = entailment.get("query", "ENTAILMENT")
    answer = entailment.get("response", "ENTAILMENT")
    claim = response.get(
        "label", "SUPPORTS" if query == "ENTAILMENT" else "REFUTES"
    )
    return consistency, query, answer, claim


def repair_required(response):
    consistency, query, answer, claim = _signals(response)
    return (
        consistency == "CONFLICT"
        or query == "NEUTRAL"
        or answer in {"CONTRADICTION", "NEUTRAL"}
        or (claim == "SUPPORTS" and query == "CONTRADICTION")
        or (claim == "REFUTES" and query == "ENTAILMENT")
    )


def paired_contexts(patcher, question, response):
    budgets = (patcher.latency_budget, patcher.vram_budget)
    if any(
        value is None or not np.isfinite(value) or value <= 0
        for value in budgets
    ):
        raise ValueError("Both context budgets must be positive and finite.")

    consistency, query, answer, _ = _signals(response)
    failure_label, _ = patcher.get_failure_label(response)
    needs_repair = repair_required(response)
    if needs_repair != (failure_label != "NO_FAILURE"):
        raise ValueError("Shared repair trigger differs from upstream diagnosis.")

    full = np.asarray(
        patcher.get_context(
            question, failure_label, consistency,
            query, answer, response["latency"],
        ),
        dtype=float,
    )
    if full.ndim != 1:
        raise ValueError("Expected a one-dimensional context vector.")

    suffix_width = (
        len(patcher.consistency_map[consistency])
        + len(patcher.entailment_map[query])
        + len(patcher.entailment_map[answer])
        + 2
    )
    label_stop = full.size - suffix_width
    label_start = label_stop - len(patcher.failure_map[failure_label])
    if label_start <= 0 or not np.array_equal(
        full[label_start:label_stop],
        patcher.failure_map[failure_label],
    ):
        raise ValueError("Upstream context layout differs from the expected layout.")

    masked = full.copy()
    masked[label_start:label_stop] = 0
    return {
        "repair_required": needs_repair,
        "failure_label": failure_label,
        "with_label": full,
        "without_label": masked,
    }
