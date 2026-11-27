"""Iterative image drift pipeline: Qwen3-VL caption -> SD3 image, repeated.

One sample runs

    I0 -> T1 -> I1 -> T2 -> I2 -> ... -> T5 -> I5

then captions I5 as T6. T6 is kept for later evaluation and is not sent to SD3.
The fixed VLM prompt is the only text the captioner sees. Ground-truth object
names and attributes are stored beside the run and are never placed in a prompt.

用法见同目录 README.md。换机器时改下面的 VLM_PATH 和 SD_PATH。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd
import requests
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
# 与本脚本放在同一目录。
CSV_PATH = ROOT / "all_five_objects_sorted_by_attributes.csv"
# 图转文、文生图的本地权重目录。换机器或换模型时改这两行。
VLM_PATH = Path(os.environ.get("GENDRIFT_VLM_PATH", r"D:\bonnie\aigc\Qwen3-VL-4B-Instruct"))
SD_PATH = Path(os.environ.get("GENDRIFT_SD_PATH", r"D:\bonnie\aigc\models\stable-diffusion-3-medium"))

VLM_PROMPT = (
    "Write a concise English image-generation prompt grounded only in this image. "
    "Order facts by importance: main subjects and distinctive landmarks first; "
    "then clearly visible counts, essential colors or clothing, actions and spatial relationships; "
    "then only essential background details. Aim for 35-45 English words. "
    "Use short factual sentences, each most important fact first. "
    "Preserve distinctive objects rather than replacing them with generic scene labels. "
    "Do not invent counts or uncertain attributes. Omit illegible text, guesses, "
    "aesthetic commentary, repetition, and phrases such as 'the image depicts'. "
    "Return only the prompt as one paragraph."
)
PROMPT_STRATEGY = "concise_priority_v1"

GENERATIONS = 5
VLM_MAX_NEW_TOKENS = 128
VLM_MAX_PIXELS = 1024 * 1024
VLM_MIN_PIXELS = 256 * 256
SD_STEPS = 28
SD_GUIDANCE = 7.0
SD_MAX_SEQUENCE_LENGTH = 256
SD_RESOLUTIONS = (768, 512)
NEGATIVE_PROMPT = ""


def log(message: str, log_path: Path | None = None) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_row(image_index: int) -> pd.Series:
    frame = pd.read_csv(CSV_PATH)
    if image_index < 0 or image_index >= len(frame):
        raise IndexError(f"image index {image_index} is outside 0..{len(frame) - 1}")
    return frame.iloc[image_index]


def download_image(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0 AIGC-drift-pipeline"},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)


def read_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": []}


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def ground_truth(row: pd.Series) -> dict:
    objects = ["man", "window", "tree", "building", "shirt"]
    return {
        name: {
            "present": int(row[name]),
            "attribute_count": int(row[f"{name}_attribute_count"]),
            "attributes": "" if pd.isna(row[f"{name}_attributes"]) else str(row[f"{name}_attributes"]),
        }
        for name in objects
    }


def caption_image(image_path: Path, log_path: Path) -> str:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    log("loading Qwen3-VL-4B-Instruct", log_path)
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": "12GiB"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(VLM_PATH, local_files_only=True)
    processor.image_processor.size = {
        "shortest_edge": VLM_MIN_PIXELS,
        "longest_edge": VLM_MAX_PIXELS,
    }
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = VLM_MIN_PIXELS
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = VLM_MAX_PIXELS
    log(f"Qwen ready in {time.perf_counter() - started:.1f}s", log_path)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    try:
        device = model.device
    except Exception:
        device = torch.device("cuda:0")
    inputs = inputs.to(device)

    generate_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=VLM_MAX_NEW_TOKENS,
            do_sample=False,
        )
    trimmed = [out[len(src) :] for src, out in zip(inputs.input_ids, generated)]
    paragraph = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    paragraph = paragraph.strip()
    log(f"caption done in {time.perf_counter() - generate_started:.1f}s, {len(paragraph)} chars", log_path)

    del inputs, generated, trimmed, model, processor, image
    release_memory()
    if not paragraph:
        raise RuntimeError(f"Qwen returned an empty caption for {image_path.name}")
    paragraph, budget = fit_prompt_budget(paragraph)
    budget_path = image_path.parent / f"{image_path.stem}_caption_budget.json"
    budget_path.write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"prompt budget: {budget['after_tokens']}; shortened={budget['shortened']}", log_path)
    return paragraph


def fit_prompt_budget(text: str) -> tuple[str, dict]:
    """Keep a readable prefix that fits ALL actual SD3 tokenizer budgets."""
    from transformers import AutoTokenizer
    tokenizers = [AutoTokenizer.from_pretrained(SD_PATH / folder, local_files_only=True)
                  for folder in ("tokenizer", "tokenizer_2", "tokenizer_3")]
    names = ("clip_l", "clip_g", "t5")
    limits = (77, 77, SD_MAX_SEQUENCE_LENGTH)
    def counts(value):
        return [token_length(t, value) for t in tokenizers]
    before = counts(text)
    words = text.split()
    while words and any(n > limit for n, limit in zip(counts(" ".join(words)), limits)):
        words.pop()
    final = " ".join(words).rstrip(",;:")
    # Prefer a complete sentence if it retains most of the fitted prefix.
    import re
    endings = list(re.finditer(r"[.!?](?:\s|$)", final))
    if len(words) < len(text.split()) and endings:
        end = endings[-1].start() + 1
        if end >= len(final) * .7:
            final = final[:end]
    after = counts(final)
    if not final or any(n > limit for n, limit in zip(after, limits)):
        raise ValueError("Unable to fit caption to SD3 tokenizer budgets")
    return final, {"strategy": PROMPT_STRATEGY, "raw_caption": text,
                   "final_caption": final, "shortened": final != text,
                   "limits": dict(zip(names, limits)),
                   "before_tokens": dict(zip(names, before)),
                   "after_tokens": dict(zip(names, after)),
                   "policy": "priority prompt; remove trailing words if needed; prefer full sentence"}


def token_length(tokenizer, text: str) -> int:
    previous = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    try:
        encoded = tokenizer(text, truncation=False, add_special_tokens=True)
    finally:
        tokenizer.model_max_length = previous
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def offload_t5_blocks(encoder, device: torch.device) -> None:
    """Run one T5 block on the GPU at a time. The full encoder does not fit in 8GB."""
    encoder.encoder.embed_tokens.to(device)
    if getattr(encoder, "shared", None) is not None:
        encoder.shared.to(device)
    encoder.encoder.final_layer_norm.to(device)

    def pre_hook(module, args, kwargs):
        module.to(device)
        args = tuple(arg.to(device) if torch.is_tensor(arg) else arg for arg in args)
        kwargs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in kwargs.items()}
        return args, kwargs

    def post_hook(module, _args, output):
        module.to("cpu")
        return output

    for block in encoder.encoder.block:
        block.to("cpu")
        block.register_forward_pre_hook(pre_hook, with_kwargs=True)
        block.register_forward_hook(post_hook)


def encode_prompt(log_path: Path, prompt: str) -> dict:
    from diffusers import StableDiffusion3Pipeline

    log("loading SD3 text encoders", log_path)
    started = time.perf_counter()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD_PATH,
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    device = torch.device("cuda")
    log(f"text encoders ready in {time.perf_counter() - started:.1f}s", log_path)

    counts = {
        "clip_l_tokens": token_length(pipe.tokenizer, prompt),
        "clip_g_tokens": token_length(pipe.tokenizer_2, prompt),
        "t5_tokens": token_length(pipe.tokenizer_3, prompt),
    }
    log(
        "prompt tokens "
        f"CLIP-L={counts['clip_l_tokens']} CLIP-G={counts['clip_g_tokens']} "
        f"T5={counts['t5_tokens']} (CLIP cuts at 77, T5 cuts at {SD_MAX_SEQUENCE_LENGTH})",
        log_path,
    )

    pair = [prompt, NEGATIVE_PROMPT]
    # Keep the pipeline's saved 77-token limit, but stop the tokenizer from warning
    # while diffusers measures the untruncated length.
    pipe.tokenizer.model_max_length = 10**9
    pipe.tokenizer_2.model_max_length = 10**9
    encode_started = time.perf_counter()
    with torch.inference_mode():
        pipe.text_encoder.to(device)
        clip_l, pooled_l = pipe._get_clip_prompt_embeds(
            prompt=pair, device=device, num_images_per_prompt=1, clip_model_index=0
        )
        clip_l = clip_l.cpu()
        pooled_l = pooled_l.cpu()
        pipe.text_encoder.to("cpu")
        release_memory()

        pipe.text_encoder_2.to(device)
        clip_g, pooled_g = pipe._get_clip_prompt_embeds(
            prompt=pair, device=device, num_images_per_prompt=1, clip_model_index=1
        )
        clip_g = clip_g.cpu()
        pooled_g = pooled_g.cpu()
        # Keep text_encoder resident: SD3's T5 helper reads its dtype.
        # It stays on CPU so the 8GB card is free for T5 blocks.
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        pipe.text_encoder_2 = None
        release_memory()

        log("encoding T5 with one block on GPU at a time", log_path)
        offload_t5_blocks(pipe.text_encoder_3, device)
        t5 = pipe._get_t5_prompt_embeds(
            prompt=pair,
            num_images_per_prompt=1,
            max_sequence_length=SD_MAX_SEQUENCE_LENGTH,
            device=device,
        ).cpu()

    clip = torch.cat([clip_l, clip_g], dim=-1)
    pooled = torch.cat([pooled_l, pooled_g], dim=-1)
    clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
    combined = torch.cat([clip, t5], dim=-2)
    log(f"text encoded in {time.perf_counter() - encode_started:.1f}s", log_path)

    embeds = {
        "prompt_embeds": combined[0:1].detach().cpu(),
        "pooled_prompt_embeds": pooled[0:1].detach().cpu(),
        "negative_prompt_embeds": combined[1:2].detach().cpu(),
        "negative_pooled_prompt_embeds": pooled[1:2].detach().cpu(),
        **counts,
    }
    del pipe, clip, t5, pooled, combined
    release_memory()
    return embeds


def diffuse(embeds: dict, height: int, seed: int, log_path: Path) -> Image.Image:
    from diffusers import StableDiffusion3Pipeline

    log(f"loading SD3 transformer for {height}px", log_path)
    started = time.perf_counter()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD_PATH,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    log(f"transformer ready in {time.perf_counter() - started:.1f}s", log_path)

    def on_step(_pipeline, step, _timestep, callback_kwargs):
        if step == 0 or step == SD_STEPS - 1 or step % 7 == 0:
            log(f"denoise step {step + 1}/{SD_STEPS}", log_path)
        return callback_kwargs

    generator = torch.Generator(device="cuda").manual_seed(seed)
    dtype = pipe.transformer.dtype
    denoise_started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt_embeds=embeds["prompt_embeds"].to(device="cuda", dtype=dtype),
            negative_prompt_embeds=embeds["negative_prompt_embeds"].to(device="cuda", dtype=dtype),
            pooled_prompt_embeds=embeds["pooled_prompt_embeds"].to(device="cuda", dtype=dtype),
            negative_pooled_prompt_embeds=embeds["negative_pooled_prompt_embeds"].to(device="cuda", dtype=dtype),
            num_inference_steps=SD_STEPS,
            guidance_scale=SD_GUIDANCE,
            height=height,
            width=height,
            generator=generator,
            callback_on_step_end=on_step,
        ).images[0]
    log(f"image ready in {time.perf_counter() - denoise_started:.1f}s", log_path)
    del pipe
    release_memory()
    return result


def generate_image(prompt: str, seed: int, run_dir: Path, log_path: Path) -> tuple[Image.Image, dict]:
    embeds = encode_prompt(log_path, prompt)
    locked = run_dir / "resolution.txt"
    if locked.exists():
        sizes = (int(locked.read_text(encoding="utf-8").strip()),)
    else:
        sizes = SD_RESOLUTIONS

    last_error = None
    for height in sizes:
        try:
            image = diffuse(embeds, height, seed, log_path)
            locked.write_text(str(height), encoding="utf-8")
            info = {
                "height": height,
                "width": height,
                "steps": SD_STEPS,
                "guidance_scale": SD_GUIDANCE,
                "clip_l_tokens": embeds["clip_l_tokens"],
                "clip_g_tokens": embeds["clip_g_tokens"],
                "t5_tokens": embeds["t5_tokens"],
            }
            return image, info
        except RuntimeError as error:
            last_error = error
            if "out of memory" not in str(error).lower():
                raise
            log(f"{height}px ran out of GPU memory, trying a smaller size", log_path)
            release_memory()
    raise RuntimeError("SD3 ran out of memory at every configured resolution") from last_error


def run(image_index: int) -> Path:
    row = load_row(image_index)
    image_id = int(row["image_id"])
    url = str(row["url"])
    seed = image_id
    run_dir = ROOT / "runs_concise" / f"single_{image_id}"
    source_dir = ROOT / "data" / "images"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    source = source_dir / f"{image_id}.jpg"
    log(f"sample image_id={image_id} index={image_index}", log_path)
    download_image(url, source)
    original = run_dir / "I0.jpg"
    if not original.exists():
        shutil.copyfile(source, original)

    config = {
        "prompt_strategy": PROMPT_STRATEGY,
        "prompt_token_limits": {"clip_l": 77, "clip_g": 77, "t5": SD_MAX_SEQUENCE_LENGTH},
        "image_id": image_id,
        "image_index": image_index,
        "url": url,
        "seed": seed,
        "generations": GENERATIONS,
        "vlm": str(VLM_PATH),
        "sd": str(SD_PATH),
        "vlm_prompt": VLM_PROMPT,
        "vlm_do_sample": False,
        "vlm_max_new_tokens": VLM_MAX_NEW_TOKENS,
        "vlm_max_pixels": VLM_MAX_PIXELS,
        "sd_steps": SD_STEPS,
        "sd_guidance_scale": SD_GUIDANCE,
        "sd_negative_prompt": NEGATIVE_PROMPT,
        "sd_max_sequence_length": SD_MAX_SEQUENCE_LENGTH,
        "sd_resolutions_try": list(SD_RESOLUTIONS),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = read_meta(run_dir / "meta.json")
    meta.update(
        {
            "image_id": image_id,
            "url": url,
            "seed": seed,
            "ground_truth_not_used_in_prompts": ground_truth(row),
        }
    )
    write_meta(run_dir / "meta.json", meta)

    for generation in range(1, GENERATIONS + 1):
        caption_path = run_dir / f"T{generation}.txt"
        image_path = run_dir / f"I{generation}.png"
        source_image = run_dir / ("I0.jpg" if generation == 1 else f"I{generation - 1}.png")

        if caption_path.exists() and caption_path.stat().st_size > 0:
            log(f"T{generation} already exists", log_path)
        else:
            started = time.perf_counter()
            paragraph = caption_image(source_image, log_path)
            caption_path.write_text(paragraph + "\n", encoding="utf-8")
            meta["stages"].append(
                {
                    "stage": "caption",
                    "name": caption_path.name,
                    "source": source_image.name,
                    "seconds": round(time.perf_counter() - started, 1),
                    "chars": len(paragraph),
                }
            )
            write_meta(run_dir / "meta.json", meta)
            log(f"wrote {caption_path.name}", log_path)

        if image_path.exists() and image_path.stat().st_size > 0:
            log(f"I{generation} already exists", log_path)
            continue

        prompt = caption_path.read_text(encoding="utf-8").strip()
        started = time.perf_counter()
        image, info = generate_image(prompt, seed, run_dir, log_path)
        image.save(image_path)
        info.update(
            {
                "stage": "diffuse",
                "name": image_path.name,
                "source": caption_path.name,
                "seconds": round(time.perf_counter() - started, 1),
                "seed": seed,
            }
        )
        meta["stages"].append(info)
        write_meta(run_dir / "meta.json", meta)
        log(f"wrote {image_path.name}", log_path)

    final_caption = run_dir / f"T{GENERATIONS + 1}.txt"
    final_image = run_dir / f"I{GENERATIONS}.png"
    if final_caption.exists() and final_caption.stat().st_size > 0:
        log(f"{final_caption.name} already exists", log_path)
    else:
        started = time.perf_counter()
        paragraph = caption_image(final_image, log_path)
        final_caption.write_text(paragraph + "\n", encoding="utf-8")
        meta["stages"].append(
            {
                "stage": "caption",
                "name": final_caption.name,
                "source": final_image.name,
                "seconds": round(time.perf_counter() - started, 1),
                "chars": len(paragraph),
                "used_for_diffusion": False,
            }
        )
        write_meta(run_dir / "meta.json", meta)
        log(f"wrote {final_caption.name}", log_path)

    log(f"finished {run_dir}", log_path)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="对 CSV 中的一行做 5 代图文迭代。")
    # 唯一的命令行参数。默认 0，即 CSV 第一行。用法见 README.md。
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="CSV 行号，从 0 开始。输出写到 runs_concise/single_<image_id>/",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this interpreter")
    try:
        run(args.image_index)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
