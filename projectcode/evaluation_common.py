"""Shared offline evaluation utilities; no generation files are modified."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT = Path(__file__).resolve().parent
CLIP_PATH = r'D:\bonnie\aigc\models\clip-vit-large-patch14'
DINO_PATH = r'D:\bonnie\aigc\models\grounding-dino-base'
CATEGORIES = ('man', 'window', 'tree', 'building', 'shirt')


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/single_497948')
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return p


def device(name):
    import torch
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name


def images(run_dir):
    found = {}
    for p in run_dir.iterdir():
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.stem.startswith('I') and p.stem[1:].isdigit():
            index = int(p.stem[1:])
            if index in found:
                raise ValueError(f'Duplicate image generation: I{index}')
            found[index] = p
    if 0 not in found:
        raise FileNotFoundError(f'No I0 image in {run_dir}')
    return sorted(found.items())


def save(output, stem, rows, metadata):
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{stem}.json').write_text(json.dumps({'metadata': metadata, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    if rows:
        with (output / f'{stem}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Saved {output / stem} (.json / .csv)', flush=True)


def versions():
    import torch
    import transformers
    return {'torch': torch.__version__, 'transformers': transformers.__version__}


def retention(reference, detected):
    ref, pred = set(reference), set(detected)
    tp = len(ref & pred)
    precision = tp / len(pred) if pred else None
    recall = tp / len(ref) if ref else None
    f1 = 2 * tp / (len(ref) + len(pred)) if ref or pred else None
    return {'retained_count': tp, 'reference_count': len(ref), 'detected_count': len(pred),
            'orr': recall, 'category_precision_vs_original': precision,
            'category_recall_vs_original': recall, 'category_f1_vs_original': f1}
