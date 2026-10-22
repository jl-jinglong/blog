"""Render inspected figure regions from the RL papers without page prose.

Usage:
    python tools/rl-figure-assets.py <pdf_dir> <output_dir>

The PDFs are downloaded from arXiv rather than committed; each crop is recorded
in sources.json together with the source URL, PDF page and SHA-256 of the file
it was measured against.
"""

import argparse
import hashlib
import json
import pathlib

import pymupdf
from PIL import Image

# name, pdf, page, figure number, crop box in PDF points
FIGURES = [
    ("ppo-clip", "ppo.pdf", 3, 1, (132.0, 446.0, 452.0, 600.0)),
    ("dpo-rlhf-vs-dpo", "dpo.pdf", 2, 1, (104.0, 68.0, 508.0, 156.0)),
    ("dpo-reward-kl", "dpo.pdf", 7, 2, (104.0, 68.0, 508.0, 226.0)),
    ("grpo-ppo-vs-grpo", "grpo.pdf", 13, 4, (66.0, 80.0, 522.0, 276.0)),
    ("dapo-clip-higher", "dapo.pdf", 3, 2, (66.0, 96.0, 546.0, 258.0)),
    ("dapo-metrics", "dapo.pdf", 10, 7, (66.0, 96.0, 546.0, 405.0)),
    ("gspo-training-curves", "gsmo.pdf", 5, 1, (66.0, 186.0, 529.0, 453.0)),
    ("gspo-clipping-fraction", "gsmo.pdf", 5, 2, (88.0, 626.0, 508.0, 744.0)),
]

SOURCES = {
    "ppo.pdf": "1707.06347",
    "dpo.pdf": "2305.18290",
    "grpo.pdf": "2402.03300",
    "dapo.pdf": "2503.14476",
    "gsmo.pdf": "2507.18071",
}

# Match the RoPE cover canvas without cropping the GRPO comparison diagram.
COVER_SIZE = (1059, 661)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--zoom", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name, filename, page_number, figure_number, box in FIGURES:
        source = args.pdf_dir / filename
        with pymupdf.open(source) as doc:
            page = doc[page_number - 1]
            clip = pymupdf.Rect(box)
            if not page.rect.contains(clip):
                raise ValueError(f"Crop outside page: {name} {box}")
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(args.zoom, args.zoom), clip=clip, alpha=False
            )
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image.save(args.output_dir / f"{name}.webp", quality=94, method=6)
        records.append(
            {
                "asset": f"{name}.webp",
                "source": f"https://arxiv.org/pdf/{SOURCES[filename]}",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "pdf_page": page_number,
                "figure": figure_number,
                "crop_points": list(box),
                "pixels": list(image.size),
                "note": "Cropped from the original figure; the caption is cited in the article.",
            }
        )
        print(f"{name}.webp: {image.size}")

    with Image.open(args.output_dir / "grpo-ppo-vs-grpo.webp") as image:
        image.thumbnail((COVER_SIZE[0] - 40, COVER_SIZE[1] - 40), Image.Resampling.LANCZOS)
        cover = Image.new("RGB", COVER_SIZE, "white")
        cover.paste(image, ((COVER_SIZE[0] - image.width) // 2, (COVER_SIZE[1] - image.height) // 2))
        cover.save(args.output_dir / "rl-cover.webp", quality=94, method=6)
    records.append(
        {
            "asset": "rl-cover.webp",
            "derived_from": "grpo-ppo-vs-grpo.webp",
            "pixels": list(COVER_SIZE),
            "operation": "Aspect-preserving resize and white padding; no cropping.",
        }
    )

    (args.output_dir / "sources.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(records)} records to sources.json")


if __name__ == "__main__":
    main()
