"""Sequential generation / evaluation of CSV samples, isolated per subprocess."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def complete(run_dir):
    expected = ['I0.jpg', 'meta.json', 'config.json']
    expected += [f'I{i}.png' for i in range(1, 6)]
    expected += [f'T{i}.txt' for i in range(1, 7)]
    return all((run_dir / f).is_file() and (run_dir / f).stat().st_size > 0 for f in expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', type=int, default=0, help='inclusive CSV row index')
    p.add_argument('--stop', type=int, default=None, help='exclusive CSV row index; default all')
    p.add_argument('--mode', choices=['all', 'generate', 'evaluate'], default='all')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--vlm-path', default=r'D:\bonnie\aigc\Qwen3-VL-4B-Instruct')
    p.add_argument('--sd-path', default=None, help='generation requires explicit existing SD3 directory')
    p.add_argument('--clip-path', default=r'D:\bonnie\aigc\models\clip-vit-large-patch14')
    p.add_argument('--dino-path', default=r'D:\bonnie\aigc\models\grounding-dino-base')
    p.add_argument('--box-threshold', type=float, default=0.35)
    p.add_argument('--text-threshold', type=float, default=0.25)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    with (ROOT / 'all_five_objects_sorted_by_attributes.csv').open(encoding='utf-8-sig', newline='') as f:
        samples = list(csv.DictReader(f))
    stop = len(samples) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(samples):
        p.error(f'require 0 <= start < stop <= {len(samples)}')
    if not all(0 <= v <= 1 for v in (args.box_threshold, args.text_threshold)):
        p.error('thresholds must be within [0, 1]')
    needs_generation = args.mode != 'evaluate' and any(
        not complete(ROOT / 'runs' / f"single_{row['image_id']}") for row in samples[args.start:stop])
    if needs_generation and not args.sd_path:
        p.error('provide --sd-path with your actual SD3 model directory')
    if not args.dry_run:
        required = []
        if needs_generation:
            required += [args.vlm_path, args.sd_path]
        if args.mode != 'generate':
            required += [args.clip_path, args.dino_path]
        for path in required:
            if not Path(path).is_dir():
                p.error(f'model directory not found: {path}')
    print(f'CSV samples={len(samples)}, selected=[{args.start}, {stop}), mode={args.mode}', flush=True)
    report_dir = ROOT / 'runs' / 'batch'
    if not args.dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index in range(args.start, stop):
        row = samples[index]
        image_id = row['image_id']
        run_dir = ROOT / 'runs' / f'single_{image_id}'
        stages = []
        if args.mode != 'evaluate' and not complete(run_dir):
            stages.append(('generate', ['run_iterative.py', '--image-index', str(index)]))
        if args.mode != 'generate':
            stages += [('clip', ['evaluate_clip.py', '--run-dir', str(run_dir), '--device', args.device,
                                 '--model-path', args.clip_path]),
                       ('dino', ['evaluate_dino.py', '--run-dir', str(run_dir), '--device', args.device,
                                 '--model-path', args.dino_path, '--box-threshold', str(args.box_threshold),
                                 '--text-threshold', str(args.text_threshold)])]
        print(f'[{index + 1}/{stop}] image_id={image_id}; stages={[s for s,_ in stages]}', flush=True)
        if args.dry_run:
            continue
        status = {'index': index, 'image_id': image_id, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'stages': []}
        env = os.environ.copy()
        env['GENDRIFT_VLM_PATH'] = args.vlm_path
        if args.sd_path:
            env['GENDRIFT_SD_PATH'] = args.sd_path
        sample_failed = False
        for name, command in stages:
            if name != 'generate' and not complete(run_dir):
                status['stages'].append({'stage': name, 'status': 'blocked_incomplete_generation'})
                sample_failed = True
                break
            log = report_dir / f'{image_id}_{name}.log'
            with log.open('w', encoding='utf-8') as handle:
                result = subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT)
            status['stages'].append({'stage': name, 'returncode': result.returncode, 'log': str(log)})
            print(f'  {name}: exit={result.returncode}; log={log}', flush=True)
            if result.returncode:
                sample_failed = True
                if name == 'generate':
                    break
        failed += int(sample_failed)
        status['status'] = 'failed' if sample_failed else 'ok'
        with (report_dir / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(status, ensure_ascii=False) + '\n')
    print(f'Finished; failed samples={failed}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
