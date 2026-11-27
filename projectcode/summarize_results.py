"""Aggregate complete paired runs without model dependencies (Python stdlib only)."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
METRICS = {'clip': ['image_vs_original', 'image_vs_previous', 'image_vs_prompt'], 'dino': ['orr']}


def interval(values, rng, repetitions):
    if len(values) < 2:
        return None, None
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return means[int(.025 * (repetitions - 1))], means[int(.975 * (repetitions - 1))]


def write(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'summary')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('--bootstrap must be >= 100')
    rng = random.Random(args.seed)
    audit, raw, stats, paired = [], [], [], []
    cohorts = {}
    for kind, metrics in METRICS.items():
        accepted = []
        for run in sorted(args.runs_dir.glob('single_*')):
            path = run / 'evaluation' / f'{kind}_metrics.json'
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                rows = payload['rows']
                indexed = {int(r['generation']): r for r in rows}
                if len(indexed) != len(rows) or set(indexed) != set(range(6)):
                    raise ValueError('requires exactly generations 0..5')
                for generation, row in indexed.items():
                    for metric in metrics:
                        value = row.get(metric)
                        if value is None and kind == 'clip' and generation == 0 and metric != 'image_vs_original':
                            continue
                        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                            raise ValueError(f'invalid {metric} at generation {generation}')
                metadata = payload['metadata']
                # Do not silently pool different evaluators, thresholds or policies.
                signature = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
                accepted.append((run.name, indexed, signature))
                audit.append({'sample': run.name, 'evaluator': kind, 'status': 'included', 'detail': ''})
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit.append({'sample': run.name, 'evaluator': kind, 'status': 'excluded', 'detail': str(error)})
        groups = sorted({s for _, _, s in accepted})
        for number, signature in enumerate(groups, 1):
            group = f'{kind}_{number}'
            samples = [(name, rows) for name, rows, s in accepted if s == signature]
            cohorts[group] = {'metadata': json.loads(signature), 'samples': [name for name,_ in samples]}
            for name, rows in samples:
                for generation, row in sorted(rows.items()):
                    for metric in metrics:
                        raw.append({'group': group, 'sample': name, 'generation': generation,
                                    'metric': metric, 'value': row.get(metric),
                                    'text_truncated': row.get('text_truncated')})
            for metric in metrics:
                for generation in range(6):
                    values = [rows[generation][metric] for _, rows in samples if rows[generation].get(metric) is not None]
                    if not values:
                        continue
                    low, high = interval(values, rng, args.bootstrap)
                    stats.append({'group': group, 'generation': generation, 'metric': metric, 'n': len(values),
                                  'mean': statistics.mean(values), 'sd': statistics.stdev(values) if len(values)>1 else None,
                                  'ci95_low': low, 'ci95_high': high})
                deltas = [rows[5][metric] - rows[1][metric] for _, rows in samples]
                low, high = interval(deltas, rng, args.bootstrap)
                paired.append({'group': group, 'metric': metric, 'comparison': 'I5_minus_I1', 'n': len(deltas),
                               'mean_change': statistics.mean(deltas), 'ci95_low': low, 'ci95_high': high,
                               'fraction_decreased': sum(x < 0 for x in deltas)/len(deltas)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir/'all_metrics.csv', raw, ['group','sample','generation','metric','value','text_truncated'])
    write(args.output_dir/'generation_summary.csv', stats, ['group','generation','metric','n','mean','sd','ci95_low','ci95_high'])
    write(args.output_dir/'paired_changes.csv', paired, ['group','metric','comparison','n','mean_change','ci95_low','ci95_high','fraction_decreased'])
    write(args.output_dir/'audit.csv', audit, ['sample','evaluator','status','detail'])
    (args.output_dir/'cohorts.json').write_text(json.dumps(cohorts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {args.output_dir}; included evaluations={sum(r["status"]=="included" for r in audit)}')
    print('Bootstrap intervals resample samples, not generations; n<2 has no CI. Groups may have different cohorts.')


if __name__ == '__main__':
    main()
