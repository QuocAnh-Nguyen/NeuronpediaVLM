#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Equivalence gate: our multimodal forward vs HuggingFace's own LLaVA forward.

Two invariants matter, and both are checked here:

1. The residual stream the lens records (block output of the last layer, pre-norm) and
   the logits we read out of it must equal HuggingFace's own hidden states / logits on
   identical inputs. ``unembed(forward_residual(batch)) == hf_model(...).logits``.
2. The stack output (``last_hidden_state``, post final norm) must also match HF's.

Run on the tiny CPU fixture during development and on the real checkpoint before any
fit on the H100:

    python scripts/check_equivalence.py --backend tiny
    python scripts/check_equivalence.py --backend hf-llava --model llava-hf/llava-1.5-7b-hf \
        --device cuda --dtype bfloat16 --image /path/to.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_lens.models.llava import LlavaLensModel  # noqa: E402
from vlm_lens.models.tiny_llava import build_tiny_llava, random_image  # noqa: E402


def build_model(args: argparse.Namespace) -> LlavaLensModel:
    if args.backend == "tiny":
        model, processor = build_tiny_llava()
        return LlavaLensModel(model, processor)
    return LlavaLensModel.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device=args.device,
        local_files_only=args.local_files_only,
    )


def compare(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float) -> bool:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    max_abs = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=1e-3)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: n={a.numel()} "
        f"max|diff|={max_abs:.3e} (atol={atol:g})"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tiny", "hf-llava"], default="tiny")
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--image", action="append", default=None, help="image path (repeatable)")
    parser.add_argument("--n-images", type=int, default=1, help="random images (tiny backend)")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model = build_model(args)

    if args.image:
        images: list = list(args.image)
    else:
        images = [random_image(seed=i) for i in range(args.n_images)]
    prompt = " ".join(["USER:", "<image>"] * len(images)) + "\nDescribe this image.\nASSISTANT:"

    batch = model.encode_mm(prompt, images)
    print(
        f"encoded: seq_len={batch.seq_len} image_tokens={batch.n_image_tokens} "
        f"n_images={0 if batch.pixel_values is None else batch.pixel_values.shape[0]} "
        f"pixel_dtype={None if batch.pixel_values is None else batch.pixel_values.dtype}"
    )

    atol = args.atol if args.backend == "hf-llava" else 0.0
    with torch.no_grad():
        ours_stack = model.forward_mm(batch)
        ours_residual = model.forward_residual(batch)
        ours_logits = model.unembed(ours_residual)

        hf_out = model.hf_model(**batch.hf_kwargs(), use_cache=False, output_hidden_states=True)
        hf_stack = hf_out.hidden_states[-1]
        hf_logits = hf_out.logits

    results = [
        compare("LM stack output (post final norm)", ours_stack, hf_stack, atol=atol),
        compare("lens readout: unembed(residual) vs hf logits", ours_logits, hf_logits, atol=atol),
    ]

    # Text-only path (upstream JacobianLens.apply and the S1 control fit use it).
    text_batch = model.encode_mm("The capital of France is", None)
    with torch.no_grad():
        text_logits = model.unembed(model.forward_residual(text_batch))
        hf_text_logits = model.hf_model(**text_batch.hf_kwargs(), use_cache=False).logits
    results.append(compare("text-only logits", text_logits, hf_text_logits, atol=atol))

    ok = all(results)
    print("EQUIVALENCE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
