"""Render inspected figure regions from official papers without page prose."""

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf
from PIL import Image


FIGURES = [
    ("qwen1-lineage", "qwen1.pdf", "2309.16609", 3, 1, (118, 374, 489, 533)),
    ("gqa-heads", "gqa.pdf", "2305.13245", 2, 2, (96, 72, 506, 211)),
    ("dca-attention", "dca.pdf", "2402.17463", 4, 2, (98, 64, 496, 203)),
    ("qwen3-post-training", "qwen3.pdf", "2505.09388", 9, 1, (70, 564, 525, 749)),
    ("qwen3-thinking-budget", "qwen3.pdf", "2505.09388", 20, 2, (70, 396, 526, 693)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, filename, arxiv, page_number, figure_number, box in FIGURES:
        source = args.pdf_dir / filename
        with pymupdf.open(source) as doc:
            page = doc[page_number - 1]
            clip = pymupdf.Rect(box)
            if not page.rect.contains(clip):
                raise ValueError(f"Crop outside page: {name}")
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=clip, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.save(args.output_dir / f"{name}.webp", quality=94, method=6)
            records.append({
                "asset": f"{name}.webp",
                "source": f"https://arxiv.org/pdf/{arxiv}",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "pdf_page": page_number,
                "figure": figure_number,
                "crop_points": list(box),
                "pixels": list(image.size),
                "note": "Full diagram retained; original caption is cited separately in the article.",
            })
    # Match the RoPE cover canvas without cropping any part of the text-model pipeline.
    with Image.open(args.output_dir / "qwen3-post-training.webp") as image:
        image.thumbnail((1003, 605), Image.Resampling.LANCZOS)
        cover = Image.new("RGB", (1059, 661), "white")
        cover.paste(image, ((1059 - image.width) // 2, (661 - image.height) // 2))
        cover.save(args.output_dir / "qwen-cover.webp", quality=94, method=6)
    records.append({"asset": "qwen-cover.webp", "derived_from": "qwen3-post-training.webp", "pixels": [1059, 661], "operation": "Aspect-preserving resize and white padding; no cropping."})
    (args.output_dir / "sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
