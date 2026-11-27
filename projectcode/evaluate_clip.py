"""Raw cosine similarities for an existing GenDrift run."""
from evaluation_common import CLIP_PATH, device, images, parser, save, versions


class ClipEvaluator:
    def __init__(self, path, target):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = target
        self.processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(path, local_files_only=True).to(target).eval()
        self.limit = self.model.config.text_config.max_position_embeddings

    def image(self, path):
        from PIL import Image
        with Image.open(path) as source:
            inputs = self.processor(images=source.convert('RGB'), return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            # Explicit projection avoids version-dependent get_*_features return types.
            pooled = self.model.vision_model(**inputs).pooler_output
            vector = self.model.visual_projection(pooled).float()
            return self.torch.nn.functional.normalize(vector, dim=-1)

    def text(self, text):
        tokenizer = self.processor.tokenizer
        count = len(tokenizer(text, truncation=False)['input_ids'])
        inputs = tokenizer(text, truncation=True, max_length=self.limit, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            pooled = self.model.text_model(**inputs).pooler_output
            vector = self.model.text_projection(pooled).float()
            vector = self.torch.nn.functional.normalize(vector, dim=-1)
        return vector, count


def cosine(a, b):
    return float((a * b).sum().item())


def main():
    p = parser(__doc__)
    p.add_argument('--model-path', default=CLIP_PATH)
    args = p.parse_args()
    paths = images(args.run_dir)
    target = device(args.device)
    evaluator = ClipEvaluator(args.model_path, target)
    vectors = {}
    rows = []
    for generation, path in paths:
        print(f'CLIP {path.name}', flush=True)
        vector = evaluator.image(path)
        vectors[generation] = vector
        caption = args.run_dir / f'T{generation}.txt'
        row = {'generation': generation, 'image': path.name,
               'image_vs_original': cosine(vector, vectors[0]),
               'image_vs_previous': cosine(vector, vectors[generation - 1]) if generation - 1 in vectors else None,
               'image_vs_prompt': None, 'prompt_file': None, 'prompt_tokens': None,
               'text_limit': evaluator.limit, 'text_truncated': None}
        if generation > 0 and caption.exists():
            text, count = evaluator.text(caption.read_text(encoding='utf-8').strip())
            row.update(image_vs_prompt=cosine(vector, text), prompt_file=caption.name,
                       prompt_tokens=count, text_truncated=count > evaluator.limit)
        rows.append(row)
    save(args.output_dir or args.run_dir / 'evaluation', 'clip_metrics', rows,
         {**versions(), 'model_path': args.model_path, 'device': target,
          'metric': 'raw cosine of normalized projected embeddings',
          'text_policy': 'truncate to model maximum; It compared with Tt; I0 has no prompt'})


if __name__ == '__main__':
    main()
