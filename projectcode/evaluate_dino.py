"""Category retention estimates, with raw boxes for human review."""
import inspect
import json
from evaluation_common import CATEGORIES, DINO_PATH, device, images, parser, retention, save, versions


class DinoEvaluator:
    def __init__(self, path, target, box_threshold=0.35, text_threshold=0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = target
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(target).eval()

    def detect(self, image):
        detections = []
        # One category per query avoids ambiguous phrase-to-category assignment.
        for category in CATEGORIES:
            inputs = self.processor(images=image, text=f'{category}.', return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            post = self.processor.post_process_grounded_object_detection
            threshold_key = 'box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
            result = post(outputs, input_ids=inputs.input_ids,
                          target_sizes=[(image.height, image.width)],
                          text_threshold=self.text_threshold,
                          **{threshold_key: self.box_threshold})[0]
            for score, box in zip(result['scores'].detach().cpu().tolist(), result['boxes'].detach().cpu().tolist()):
                detections.append({'category': category, 'score': score, 'box_xyxy': box})
        return detections


def main():
    p = parser(__doc__)
    p.add_argument('--model-path', default=DINO_PATH)
    p.add_argument('--box-threshold', type=float, default=0.35)
    p.add_argument('--text-threshold', type=float, default=0.25)
    args = p.parse_args()
    for threshold in (args.box_threshold, args.text_threshold):
        if not 0 <= threshold <= 1:
            p.error('thresholds must be within [0, 1]')
    from PIL import Image, ImageDraw
    paths = images(args.run_dir)
    meta = json.loads((args.run_dir / 'meta.json').read_text(encoding='utf-8'))
    ground_truth = meta['ground_truth_not_used_in_prompts']
    reference = [c for c in CATEGORIES if int(ground_truth[c]['present']) == 1]
    target = device(args.device)
    evaluator = DinoEvaluator(args.model_path, target, args.box_threshold, args.text_threshold)
    output = args.output_dir or args.run_dir / 'evaluation'
    (output / 'dino_boxes').mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for generation, path in paths:
        print(f'DINO {path.name}', flush=True)
        with Image.open(path) as source:
            image = source.convert('RGB')
        detections = evaluator.detect(image)
        detected = sorted({d['category'] for d in detections})
        rows.append({'generation': generation, 'image': path.name,
                     'detected_categories': '|'.join(detected), **retention(reference, detected)})
        records.append({'generation': generation, 'image': path.name, 'detections': detections})
        draw = ImageDraw.Draw(image)
        for d in detections:
            box = d['box_xyxy']
            draw.rectangle(box, outline='red', width=2)
            draw.text((box[0], box[1]), f"{d['category']} {d['score']:.2f}", fill='red')
        image.save(output / 'dino_boxes' / f'I{generation}.jpg')
    save(output, 'dino_metrics', rows, {**versions(), 'model_path': args.model_path,
         'device': target, 'box_threshold': args.box_threshold, 'text_threshold': args.text_threshold,
         'reference_categories': reference, 'query_policy': 'one category per query',
         'scope': 'five-category presence vs original CSV, not detector accuracy or instance matching',
         'undefined_ratios': 'null; F1 is zero when only one set is empty'})
    (output / 'dino_detections.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
