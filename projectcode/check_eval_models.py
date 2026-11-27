"""Verify offline loading AND inference, sequentially to limit GPU memory."""
import gc
import json
from pathlib import Path
import traceback
from evaluation_common import CLIP_PATH, DINO_PATH, device, parser, versions


def main():
    p = parser(__doc__)
    p.add_argument('--clip-path', default=CLIP_PATH)
    p.add_argument('--dino-path', default=DINO_PATH)
    p.add_argument('--image', type=Path, default=None)
    args = p.parse_args()
    target = device(args.device)
    image_path = args.image or args.run_dir / 'I0.jpg'
    if not image_path.is_file():
        p.error(f'Image not found: {image_path}')
    import torch
    from PIL import Image
    report = {**versions(), 'device': target, 'image': str(image_path), 'models': {}}
    for name, path in (('clip', args.clip_path), ('dino', args.dino_path)):
        evaluator = None
        try:
            print(f'Checking {name}: {path}', flush=True)
            if name == 'clip':
                from evaluate_clip import ClipEvaluator, cosine
                evaluator = ClipEvaluator(path, target)
                vector = evaluator.image(image_path)
                text, count = evaluator.text('A photograph of a street with buildings.')
                result = {'cosine': cosine(vector, text), 'text_tokens': count,
                          'image_embedding_shape': list(vector.shape)}
            else:
                from evaluate_dino import DinoEvaluator
                evaluator = DinoEvaluator(path, target)
                with Image.open(image_path) as source:
                    detections = evaluator.detect(source.convert('RGB'))
                result = {'detection_count': len(detections), 'detections': detections}
            report['models'][name] = {'status': 'ok', 'path': path, **result}
            print(f'{name}: offline loading and inference OK', flush=True)
        except Exception as error:
            report['models'][name] = {'status': 'failed', 'path': path,
                                      'error': str(error), 'traceback': traceback.format_exc()}
            print(f'{name}: FAILED: {error}', flush=True)
        finally:
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    output = args.output_dir or args.run_dir / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'model_check.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {destination}', flush=True)
    if any(r['status'] != 'ok' for r in report['models'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
