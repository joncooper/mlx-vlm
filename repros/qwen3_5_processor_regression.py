#!/usr/bin/env python3

import argparse
import importlib.metadata
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")


def show_load_processor_source_hint(model: str) -> None:
    utils_path = Path(__file__).resolve().parents[1] / "mlx_vlm" / "utils.py"
    if not utils_path.exists():
        return
    for line in utils_path.read_text(encoding="utf-8").splitlines():
        if "AutoProcessor.from_pretrained" in line and "use_fast=True" in line:
            print("\nmlx_vlm source hint:")
            print(f"  {utils_path}: {line.strip()}")
            print(f"  This matches the failing fast path for model={model}")
            return


def try_process(label: str, model: str, use_fast: bool) -> bool:
    print(f"\n== {label} ==")
    try:
        processor = AutoProcessor.from_pretrained(
            model,
            use_fast=use_fast,
            trust_remote_code=True,
        )
    except Exception as exc:  # pragma: no cover - repro helper
        print(f"status: FAIL during load ({type(exc).__name__})")
        print(str(exc))
        traceback.print_exc()
        return False

    print(f"processor: {processor.__class__.__name__}")

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        print(f"image_processor: {image_processor.__class__.__name__}")

    try:
        outputs = processor(
            text=["<|image_pad|> Describe this image."],
            images=[dummy_image()],
            return_tensors="mlx",
        )
    except Exception as exc:  # pragma: no cover - repro helper
        print(f"status: FAIL ({type(exc).__name__})")
        print(str(exc))
        traceback.print_exc()
        return False

    print("status: OK")
    print(f"keys: {sorted(outputs.keys())}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the Qwen 3.5 processor failure on mlx-vlm main."
    )
    parser.add_argument("--model", required=True, help="HF repo ID or local model path")
    args = parser.parse_args()

    print("Environment:")
    print(f"  python={sys.version.split()[0]}")
    print(f"  transformers={version('transformers')}")
    print(f"  mlx-vlm={version('mlx-vlm')}")
    print(f"  torch={version('torch')}")
    print(f"  torchvision={version('torchvision')}")
    print(f"  model={args.model}")
    show_load_processor_source_hint(args.model)

    fast_ok = try_process(
        "AutoProcessor.from_pretrained(use_fast=True)",
        args.model,
        True,
    )
    slow_ok = try_process(
        "AutoProcessor.from_pretrained(use_fast=False)",
        args.model,
        False,
    )

    if fast_ok:
        print("\nUnexpected result: fast path succeeded.")
        return 1

    if not slow_ok:
        print("\nUnexpected result: slow path failed.")
        return 1

    print("\nSummary: fast path failed and slow path succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
