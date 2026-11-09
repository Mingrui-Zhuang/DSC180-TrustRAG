"""Summarize saved smoke artifacts without importing models or calling APIs."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if not rows:
        raise ValueError(f'No rows in {path}')
    correct = sum(row['EM'].lower() == 'true' for row in rows)
    return {
        'examples': len(rows), 'correct': correct,
        'upstream_em_percent': 100 * correct / len(rows),
        'mean_reported_latency_seconds': statistics.mean(float(row['latency']) for row in rows),
        'mean_reported_incremental_vram_mib': statistics.mean(float(row['vram_usage']) for row in rows),
        'diagnoses': dict(Counter(row['failure_label'] for row in rows)),
        'repair_actions': dict(Counter(row.get('action', '-1') for row in rows)),
        'predictions': [{key: row[key] for key in
                         ('question', 'response', 'gt_response', 'EM', 'failure_label')}
                        for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--budget-ledger', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_directory
    validation = run / 'smoke/files_hotpotqa_v'
    result = {
        'scope': 'Two-question HotpotQA integration test; not a full paper reproduction',
        'baseline': summarize(validation / 'failure_statistics.csv'),
        'policy_evaluation': summarize(validation / 'bandit_eval_dataset.csv'),
        'budget_reservations': json.loads(args.budget_ledger.read_text()),
        'policy_checkpoint_saved': (run / 'smoke/files_hotpotqa_t/linucb_patcher.pkl').is_file(),
        'training_audit': json.loads((run / 'training-audit.json').read_text()),
    }
    (run / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    budget = result['budget_reservations']
    cost = budget['reserved_nanodollars'] / 1_000_000_000
    lines = [
        '# D2R-RAG: initial execution results', '',
        '**Status: the small baseline, training and policy-evaluation pipeline executed. '
        'The full paper results have not been reproduced.**', '',
        'The test uses the first two HotpotQA questions per split and secondary '
        'contexts from five examples. The policy is trained for the original two epochs. '
        'The API generator is GPT-4o-mini with a test-only 512-token output cap.', '',
        '| Stage | Cases | Correct by upstream matcher | Mean reported latency | Mean incremental VRAM |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, key in [('Baseline', 'baseline'), ('Policy evaluation', 'policy_evaluation')]:
        row = result[key]
        lines.append(f"| {label} | {row['examples']} | {row['correct']}/{row['examples']} "
                     f"({row['upstream_em_percent']:.1f}%) | "
                     f"{row['mean_reported_latency_seconds']:.3f} s | "
                     f"{row['mean_reported_incremental_vram_mib']:.3f} MiB |")
    lines += ['', '## Interpretation', '',
              'These two cases cannot estimate the paper\'s 500-case validation results. '
              'The metric uses the authors\' `qa_metrics.em.em_match` function. '
              'API generation is stochastic and the corpus is reduced.', '',
              'Baseline diagnosis counts: ' + json.dumps(result['baseline']['diagnoses']) + '.',
              'Policy-evaluation diagnosis counts: ' + json.dumps(result['policy_evaluation']['diagnoses']) + '.',
              'Policy-evaluation action counts (-1 means no repair): ' +
              json.dumps(result['policy_evaluation']['repair_actions']) + '.', '',
              'A wrong answer labeled NO_FAILURE is a useful diagnostic example for '
              'TrustRAG; a two-case test cannot establish how often that occurs.', '',
              '## Spending and reproducibility', '',
              f"The persistent ledger reserved **${cost:.6f}** across **{budget['requests']}** "
              'attempted API requests, including interrupted attempts. This is a '
              'conservative maximum reservation, not a report of actual billing. '
              'The $1 cap and 40-attempt test ceiling remain in effect.', '',
              'A CPU-thread oversubscription issue was resolved by limiting the test '
              'to four threads. The wrapper also initializes a training-loop variable '
              'that can otherwise be used before assignment. Original upstream source '
              'files remain unchanged.', '',
              'Reported latency and memory use follow the original code, whose '
              'measurement stops before diagnostic checks. The 6 MiB policy budget '
              'is incremental allocation, not the GPU memory needed to load all models.', '',
              '## Remaining work', '',
              'A guarded full-size campaign runner is prepared in '
              '`reproduction/full_run.py` for the original 500 training '
              'and 500 validation cases. Agree a separate API budget before launching '
              'it, then evaluate baseline, trained LinUCB repair, relevance and '
              'faithfulness. Thompson Sampling, ablations and FEVER remain separate '
              'reproduction stages. See `reproduction/PROTOCOL.md` for settings and '
              'reference results.', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('baseline', 'policy_evaluation')}, indent=2))


if __name__ == '__main__':
    main()
