#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Equivalence gate: our multimodal forward vs HuggingFace's own forward.

Two invariants matter, and both are checked here:

1. The residual stream the lens records (block output of the last layer, pre-norm) and
   the logits we read out of it must equal HuggingFace's own hidden states / logits on
   identical inputs. ``unembed(forward_residual(batch)) == hf_model(...).logits``.
2. The stack output (``last_hidden_state``, post final norm) must also match HF's.

Run on the tiny CPU fixture during development and on the real checkpoint before any
fit on the H100:

    python scripts/check_equivalence.py --backend tiny
    python scripts/check_equivalence.py --backend hf-llava --model llava-7b \\
        --device cuda --dtype bfloat16 --image /path/to.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_lens.models.registry import MODEL_SPECS  # noqa: E402
from vlm_lens.models.tiny_llava import random_image  # noqa: E402


def build_model(args: argparse.Namespace) -> Any:
    """Wrapper for the ``--model`` family: tiny CPU fixture or the real checkpoint."""
    spec = MODEL_SPECS[args.model]
    if args.backend == "tiny":
        hf_model, processor = spec.build_tiny()
        return spec.lens_model_cls()(hf_model, processor)
    return spec.lens_model_cls().from_pretrained(
        spec.hf_id,
        dtype=getattr(torch, args.dtype),
        device=args.device,
        local_files_only=args.local_files_only,
    )


def residual_layers(model: Any, batch: Any) -> dict[int, torch.Tensor]:
    """``forward_residual`` normalized to ``{layer: pre-norm block output}``.

    Wrappers disagree on the return shape: LLaVA's returns the final-layer tensor,
    Qwen2-VL's (contract) returns a ``{layer: tensor}`` mapping for the captured
    layers. The gate checks whatever the wrapper captures.
    """
    residual = model.forward_residual(batch)
    if isinstance(residual, dict):
        return {int(layer): tensor for layer, tensor in residual.items()}
    return {len(model.layers) - 1: residual}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["tiny", "hf-llava"],
        default="tiny",
        help="tiny = CPU fixture for the selected --model; hf-llava = real checkpoint "
        "(historical name; loads whichever --model selects)",
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        default="llava-7b",
        help="checkpoint family: llava-7b (default), llava-13b or qwen2vl",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--image", action="append", default=None, help="image path (repeatable)")
    parser.add_argument("--n-images", type=int, default=1, help="random images (tiny backend)")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    model = build_model(args)

    if args.image:
        images: list = list(args.image)
    else:
        images = [random_image(seed=i) for i in range(args.n_images)]
    prompt = " ".join(["USER:", "<image>"] * len(images)) + "\nDescribe this image.\nASSISTANT:"

    batch = model.encode_mm(prompt, images)
    print(
        f"encoded: seq_len={batch.seq_len} image_tokens={batch.n_image_tokens} "
        f"n_images={len(images)} "
        f"pixel_dtype={None if batch.pixel_values is None else batch.pixel_values.dtype}"
    )

    atol = args.atol if args.backend == "hf-llava" else 0.0
    with torch.no_grad():
        ours_stack = model.forward_mm(batch)
        residuals = residual_layers(model, batch)

        hf_out = model.hf_model(**batch.hf_kwargs(), use_cache=False, output_hidden_states=True)
        hf_stack = hf_out.hidden_states[-1]
        hf_logits = hf_out.logits

    results = [
        compare("LM stack output (post final norm)", ours_stack, hf_stack, atol=atol),
    ]
    # Pre-norm block outputs HF exposes: block l's output is hidden_states[l + 1].
    # The last block's pre-norm output is hook-only, so it is checked via the logits.
    last_layer = max(residuals)
    for layer in sorted(residuals):
        if layer < last_layer:
            results.append(
                compare(
                    f"block output layer {layer} (pre norm)",
                    residuals[layer],
                    hf_out.hidden_states[layer + 1],
                    atol=atol,
                )
            )
    results.append(
        compare(
            "lens readout: unembed(residual) vs hf logits",
            model.unembed(residuals[last_layer]),
            hf_logits,
            atol=atol,
        )
    )

    # Text-only path (upstream JacobianLens.apply and the S1 control fit use it).
    text_batch = model.encode_mm("The capital of France is", None)
    with torch.no_grad():
        text_residuals = residual_layers(model, text_batch)
        text_logits = model.unembed(text_residuals[max(text_residuals)])
        hf_text_logits = model.hf_model(**text_batch.hf_kwargs(), use_cache=False).logits
    results.append(compare("text-only logits", text_logits, hf_text_logits, atol=atol))

    ok = all(results)
    print("EQUIVALENCE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
