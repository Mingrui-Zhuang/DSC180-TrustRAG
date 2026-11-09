"""Guarded original-size HotpotQA stages. Dry-run is safe and makes no calls."""
import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time


STAGES = {'baseline': 'experiments/analysis_shortanswer.py',
          'train': 'src/train_shortanswer.py',
          'repair': 'experiments/analysis_shortanswer_patched.py',
          'metrics': 'src/report_metrics.py',
          'metrics-repaired': 'src/report_metrics_patched.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--budget-usd', type=Decimal, required=True,
                        help='Total campaign allowance, shared across all stages/restarts')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.budget_usd.is_finite() or not 0 < args.budget_usd <= 100:
        parser.error('Budget must be positive and no greater than $100')
    root = Path(__file__).resolve().parents[1]
    campaign = args.campaign.resolve()
    repo = root / 'D2R-RAG'
    plan = {'stage': args.stage, 'script': STAGES[args.stage],
            'campaign': str(campaign), 'budget_usd': str(args.budget_usd),
            'training_examples': 500, 'validation_examples': 500,
            'training_epochs': 2, 'model': 'gpt-4o-mini', 'policy': 'LinUCB',
            'request_ceiling': 50000, 'max_output_tokens': 16384,
            'hash_seed': 0, 'cpu_threads': 4,
            'guard': 'Reserve maximum; settle verified successful usage; retain failed reservations',
            'code_fix': 'Initialize params_updates on each training row',
            'source_sha256': hashlib.sha256((repo / STAGES[args.stage]).read_bytes()).hexdigest()}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, *sys.argv])
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('OPENAI_API_KEY must be loaded privately before execution')
    campaign.mkdir(parents=True, exist_ok=True)
    ledger = campaign / 'budget.json'
    ceiling = int(args.budget_usd * 1_000_000_000)
    if ledger.exists() and json.loads(ledger.read_text()).get('limit_nanodollars') != ceiling:
        parser.error('Existing campaign budget differs. Do not silently reset or raise its allowance.')
    for folder in ('files_hotpotqa_t', 'files_hotpotqa_v'):
        (campaign / folder).mkdir(exist_ok=True)
    if not (campaign / 'src').exists():
        (campaign / 'src').symlink_to(repo / 'src', target_is_directory=True)
    if args.stage == 'repair' and not (campaign / 'files_hotpotqa_t/linucb_patcher.pkl').is_file():
        parser.error('Complete training before repair')
    expected = campaign / ('files_hotpotqa_v/bandit_eval_dataset.csv' if args.stage == 'metrics-repaired'
                           else 'files_hotpotqa_v/failure_statistics.csv')
    if args.stage.startswith('metrics') and not expected.is_file():
        parser.error('Required evaluation table is missing')
    import budget_guard
    import metered_guard
    budget_guard.LIMIT = ceiling
    budget_guard.MAX_REQUESTS = 50000
    budget_guard.MAX_OUTPUT = 16384  # GPT-4o-mini's original model output ceiling.
    budget_guard.MAX_COMPLETIONS = 3  # RAGAS can request multiple completions.
    metered_guard.install(ledger)
    sys.path.insert(0, str(repo))
    evaluation_batches = []
    if args.stage.startswith('metrics'):
        from src import evaluation
        original_evaluate = evaluation.evaluate

        def save_evaluation(*call_args, **call_kwargs):
            result = original_evaluate(*call_args, **call_kwargs)
            frame = result.to_pandas()
            destination = campaign / f'{args.stage}-ragas-{time.time_ns()}.csv'
            frame.to_csv(destination, index=False)
            evaluation_batches.append(frame)
            print('Saved evaluation table:', destination, flush=True)
            return result

        evaluation.evaluate = save_evaluation
    (campaign / (args.stage + '-plan.json')).write_text(json.dumps(plan, indent=2))
    os.chdir(campaign)
    source_path = repo / STAGES[args.stage]
    source = source_path.read_text()
    if args.stage == 'train':
        target = '        for row in tqdm.tqdm(dataset):\n'
        if source.count(target) != 1:
            raise RuntimeError('Review the training compatibility fix for changed source')
        source = source.replace(target, target + '            params_updates = {}\n')
    sys.argv = [str(source_path), 'hotpotqa']
    namespace = {'__name__': '__main__', '__file__': str(source_path)}
    exec(compile(source, str(source_path), 'exec'), namespace)
    if args.stage in ('baseline', 'repair'):
        import pandas as pd
        name = 'failure_statistics.csv' if args.stage == 'baseline' else 'bandit_eval_dataset.csv'
        frame = pd.read_csv(campaign / 'files_hotpotqa_v' / name, sep='\t')
        if len(frame) != 500:
            raise RuntimeError(f'Expected 500 evaluation examples, received {len(frame)}')
        summary = {'rows': len(frame), 'upstream_em_percent':
                   100 * frame['EM'].astype(str).str.lower().eq('true').mean(),
                   'mean_reported_latency_seconds': frame['latency'].mean(),
                   'mean_reported_incremental_vram_mib': frame['vram_usage'].mean(),
                   'diagnoses': frame['failure_label'].value_counts().to_dict()}
        (campaign / f'{args.stage}-summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    elif args.stage.startswith('metrics'):
        import pandas as pd
        frame = pd.concat(evaluation_batches, ignore_index=True)
        columns = ['answer_relevancy', 'faithfulness']
        summary = {'rows': len(frame), 'valid_scores': frame[columns].count().to_dict(),
                   'means': frame[columns].mean().to_dict()}
        (campaign / f'{args.stage}-summary.json').write_text(json.dumps(summary, indent=2))
        if len(frame) != 500 or frame[columns].isna().any().any():
            raise RuntimeError('Evaluation is incomplete; inspect saved RAGAS tables and the budget ledger')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
