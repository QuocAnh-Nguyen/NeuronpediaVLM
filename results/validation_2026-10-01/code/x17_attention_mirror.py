#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""X17: the cross-modal attention mirror — does the LLM's text->image-token attention
track the vision tower's CLIP attention, and do hallucinating captions look different?

Hypothesis (D44, H3 — explicitly CORRELATIONAL, no intervention): the LLM's
text->image-token attention mirrors the vision tower's CLS->patch attention onto the
same image (the DAMRO claim), and images whose captions hallucinate have different —
lower / background-shifted — cross-modal attention profiles.

Method, per COCO image (one image in memory at a time, no grad, sequence length capped
by ``--max-seq-len``):

  (a) LLM side — one prefill forward with ``output_attentions=True`` (the model is
      loaded with ``--attn-impl`` eager by default so HF materializes attention
      weights). For every fitted lens layer, the attention FROM the query position(s)
      ONTO the image-token key block ``[n_image_tokens]`` is extracted: with
      ``--query-mode last`` the query is the LAST PROMPT position (the prefill decoding
      position); with ``mean-text`` query rows are averaged over all text (non-image)
      prompt positions. Per layer: the head-averaged per-patch vector, its mean/max,
      and the attention SHARE — the fraction of the query rows' attention mass landing
      on image tokens (renormalized over unmasked keys).
  (b) CLIP side — the vision tower runs once with ``output_attentions=True``; the
      CLS-token->patch attention of the FINAL vision layer (plus optionally
      ``--clip-mid-layer``) is head-averaged into a per-patch vector. LLaVA's
      "default" feature strategy drops the tower's CLS embedding, so LLM image token i
      corresponds to CLIP patch i (both in raster order) — asserted at runtime.
  (c) Mirror correlation — per LLM layer, Spearman rho between CLIP's per-patch vector
      and the LLM's per-patch vector.
  (d) Hallucination join — the caption is generated greedily (x13's prompt/loop) and
      its category mentions are classified grounded/hallucinated with x14's matcher
      against ``instances_val2014.json``. Per image four statistics are formed
      (share at the mid fitted layer, mean share over layers, rho at the mid layer,
      mean rho over layers) and reported as group means for images WITH vs WITHOUT a
      hallucinated mention, plus point-biserial correlation and AUROC (positive class
      = hallucinating image).

Output (``--json``): per-image records (per-layer share/patch_mean/patch_max and
per-patch vectors, CLIP stats, per-layer rho, hallucinated/grounded lists), the
aggregate per-layer table, the hallucination join, a digest, provenance/meta, and the
interpretation-limits note. ``--backend tiny`` runs the whole pipeline on CPU; its
random ``t<n>`` captions match no COCO category, so the hallucinating group is empty
there and the join reports None (printed as dashes) — that is the expected smoke
behavior, not a failure.

LIMITS (embedded in the JSON as ``interpretation_limits``): correlational only. Group
differences over small n are confound-prone (image difficulty, object count/size,
caption length), and AUROC / point-biserial estimates are unstable below ~30 images.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch  # noqa: E402

import vlm_lens  # noqa: E402, F401  # installs the vendored jlens path: must precede jlens
import x14_workspace_probe  # noqa: E402  # the hallucination matcher (classify_generated_words)
from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    PROMPT_TEMPLATE,
    list_coco_images,
    prompt_template_hash,
    prompt_text,
    select_images,
)
from vlm_lens.interventions import generate_with_edits  # noqa: E402
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch  # noqa: E402

DEFAULT_LENS_DIR = "/data/anhnq/vlm-lens-out/validation/s2-merged/artifacts"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"
DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"

#: The four per-image statistics joined against the hallucination label, in report order.
STATISTICS: tuple[str, ...] = (
    "share_mid_layer",
    "share_mean_over_layers",
    "rho_mid_layer",
    "rho_mean_over_layers",
)

ATTENTION_SHARE_NOTE = (
    "share = fraction of the query row(s)' attention mass on the image-token keys, "
    "renormalized over unmasked keys (each head's rows sum to 1 without padding); "
    "'patch_mean'/'patch_max' are the mean/max of the head-averaged per-patch "
    "attention vector over the image tokens; the per-patch vector is stored as "
    "'patch_vector' in raster patch order"
)
QUERY_MODE_NOTE = {
    "last": "query = the LAST PROMPT position (the prefill decoding position)",
    "mean-text": "query rows averaged over all text (non-image) prompt positions",
}
ALIGNMENT_NOTE = (
    "LLaVA's 'default' feature strategy drops the tower's CLS embedding and keeps the "
    "576 patch embeddings in raster order, so LLM image token i corresponds to CLIP "
    "patch key i+1 of the CLS attention row; the script asserts the per-patch vector "
    "length equals the prompt's image-token count and refuses to run otherwise"
)
LIMITS_NOTE = (
    "CORRELATIONAL ONLY (D44/H3): no intervention is performed and no causal claim is "
    "supported. The mirror is measured as rank agreement (Spearman rho per layer) "
    "between CLIP CLS->patch attention and the LLM's text->image-token attention on "
    "the same image; the hallucination join compares group means, point-biserial r "
    "and AUROC (positive class = image whose caption has >=1 hallucinated mention). "
    "Confounds (image difficulty, object count/size, caption length, prompt position, "
    "layer/head choice) are uncontrolled, and AUROC/point-biserial estimates are "
    "unstable below ~30 images; treat every number here as an association, not a cause."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lens-dir", default=DEFAULT_LENS_DIR,
        help="J source lens dir: only its fitted text-layer list is used (attention is "
        "extracted 'per fitted layer')",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        help="instances_val2014.json: per-image ground-truth categories (x14 grounding)",
    )
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="image-selection seed")
    parser.add_argument("--question", default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument(
        "--backend", choices=("hf-llava", "tiny"), default="hf-llava",
        help="model backend: 'hf-llava' (default, CUDA) or the tiny CPU smoke fixture",
    )
    parser.add_argument(
        "--attn-impl", default="eager",
        help="attention implementation the model is loaded with (eager so HF returns "
        "output_attentions; the tiny backend is switched to eager too)",
    )
    parser.add_argument(
        "--query-mode", choices=("last", "mean-text"), default="last",
        help="LLM query position(s) for the text->image-token attention",
    )
    parser.add_argument(
        "--clip-mid-layer", type=int, default=None,
        help="also extract CLS->patch attention at this (mid) vision layer index",
    )
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> LlavaLensModel:
    """The mirror model: the HF checkpoint (default, CUDA, eager) or the tiny fixture."""
    if args.backend == "tiny":
        from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava

        hf_model, processor = build_tiny_llava(TinyLlavaConfig())
        hf_model.set_attn_implementation(args.attn_impl)
        return LlavaLensModel(hf_model, processor)
    return LlavaLensModel.from_pretrained(
        dtype=torch.bfloat16,
        device="cuda",
        local_files_only=True,
        attn_implementation=args.attn_impl,
    )


@torch.no_grad()
def llm_image_attention(
    model: LlavaLensModel,
    batch: MultimodalBatch,
    layers: list[int],
    query_mode: str,
) -> dict[int, dict[str, Any]]:
    """Per fitted layer: the prompt's attention onto the image-token key block.

    One prefill forward with ``output_attentions=True``; per layer the query row(s)
    (last prompt position, or the mean over text positions) are head-averaged over the
    image-token keys into a per-patch vector, summarized (mean, max, share).
    """
    output = model.hf_model.model(
        **batch.hf_kwargs(), use_cache=False, output_attentions=True, return_dict=True
    )
    # sdpa (the transformers default) returns an empty/None attentions payload when
    # output_attentions=True is requested - only eager materializes the weights.
    attentions = output.attentions
    if (
        attentions is None
        or len(attentions) <= max(layers)
        or any(weight is None for weight in attentions)
    ):
        raise RuntimeError(
            "the model returned no attention weights: reload with "
            f"--attn-impl eager (requested implementation is {model.hf_model.config._attn_implementation!r})"
        )

    image_positions = batch.image_token_mask[0].nonzero(as_tuple=True)[0]
    n_image = int(image_positions.numel())
    if n_image == 0:
        raise RuntimeError("prompt carries no image tokens")
    if int(image_positions[-1]) - int(image_positions[0]) + 1 != n_image:
        raise RuntimeError("image tokens are not one contiguous block; patch alignment would break")
    if query_mode == "last":
        query_positions = torch.tensor(
            [batch.seq_len - 1], dtype=torch.long, device=batch.input_ids.device
        )
    else:
        query_positions = (~batch.image_token_mask[0]).nonzero(as_tuple=True)[0]
        if query_positions.numel() == 0:
            raise RuntimeError("prompt carries no text positions")
    valid_keys = batch.attention_mask[0].bool().nonzero(as_tuple=True)[0]

    per_layer: dict[int, dict[str, Any]] = {}
    for layer in layers:
        attention = attentions[layer][0].float()  # [heads, seq, seq]
        rows = attention.index_select(1, query_positions).mean(dim=1)  # [heads, seq]
        patch = rows.index_select(1, image_positions)  # [heads, n_image]
        vector = patch.mean(dim=0)  # [n_image] head-averaged per-patch attention
        total = rows.index_select(1, valid_keys).sum(dim=1)  # [heads] (== 1 unpadded)
        share = float((patch.sum(dim=1) / total.clamp_min(1e-12)).mean())
        per_layer[layer] = {
            "share": share,
            "patch_mean": float(vector.mean()),
            "patch_max": float(vector.max()),
            "patch_vector": vector.cpu(),
        }
    del output
    return per_layer


@torch.no_grad()
def clip_cls_patch_attention(
    model: LlavaLensModel, batch: MultimodalBatch, mid_layer: int | None
) -> dict[str, dict[str, Any]]:
    """CLS->patch attention of the final vision layer (plus an optional mid layer).

    Returns ``{"final": record, "mid": record?}``; each record carries the
    head-averaged per-patch vector (patch keys only, CLS key dropped), its
    mean/max, and the fraction of the CLS row's attention mass on patches.
    """
    tower = model.hf_model.model.vision_tower
    vision_output = tower(
        pixel_values=batch.pixel_values, output_attentions=True, return_dict=True
    )
    # same sdpa caveat as the LLM side: only eager materializes the weights
    vision_attentions = vision_output.attentions
    if (
        vision_attentions is None
        or not vision_attentions
        or any(weight is None for weight in vision_attentions)
    ):
        raise RuntimeError("the vision tower returned no attention weights (need eager attention)")
    n_vision_layers = len(vision_attentions)
    wanted: dict[str, int] = {"final": n_vision_layers - 1}
    if mid_layer is not None:
        if not 0 <= mid_layer < n_vision_layers:
            raise ValueError(
                f"--clip-mid-layer {mid_layer} is outside the vision tower "
                f"(0..{n_vision_layers - 1})"
            )
        wanted["mid"] = mid_layer

    extracted: dict[str, dict[str, Any]] = {}
    for name, index in wanted.items():
        attention = vision_attentions[index][0].float()  # [heads, 1+patches, 1+patches]
        cls_row = attention[:, 0, :]  # [heads, 1+patches]
        patch = cls_row[:, 1:]  # drop the CLS key: patch keys only
        vector = patch.mean(dim=0)  # [n_patches] head-averaged
        extracted[name] = {
            "layer": index,
            "share_patches": float(patch.sum(dim=1).mean()),
            "patch_mean": float(vector.mean()),
            "patch_max": float(vector.max()),
            "patch_vector": vector.cpu(),
        }
    return extracted


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share the mean rank)."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        end = start
        while end + 1 < flat.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation; None when either side is constant."""
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denominator <= 0.0:
        return None
    return float((x * y).sum() / denominator)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rho = Pearson on average ranks; None for constant/short input."""
    if x.shape != y.shape or x.size < 2:
        return None
    return _pearson(_average_ranks(x), _average_ranks(y))


def _point_biserial(labels: list[bool], values: list[float]) -> float | None:
    """Pearson r between the 0/1 group label and the statistic."""
    binary = np.asarray([1.0 if label else 0.0 for label in labels], dtype=np.float64)
    return _pearson(binary, np.asarray(values, dtype=np.float64))


def _auroc(labels: list[bool], values: list[float]) -> float | None:
    """AUROC via the Mann-Whitney statistic (positive class = True = hallucinating)."""
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    if not positives or not negatives:
        return None
    ranks = _average_ranks(np.asarray(values, dtype=np.float64))
    n_pos, n_neg = len(positives), len(negatives)
    u_pos = float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _r(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _vector_json(vector: torch.Tensor) -> list[float]:
    return [round(float(component), 6) for component in vector.tolist()]


def _layer_record(record: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe per-layer LLM attention record (vectors rounded to 6 decimals)."""
    return {
        "share": _r(record["share"]),
        "patch_mean": _r(record["patch_mean"]),
        "patch_max": _r(record["patch_max"]),
        "patch_vector": _vector_json(record["patch_vector"]),
    }


def _clip_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": record["layer"],
        "share_patches": _r(record["share_patches"]),
        "patch_mean": _r(record["patch_mean"]),
        "patch_max": _r(record["patch_max"]),
        "patch_vector": _vector_json(record["patch_vector"]),
    }


def _cell(value: float | None, width: int, precision: int = 4) -> str:
    return f"{value:>{width}.{precision}f}" if value is not None else f"{'-':>{width}}"


def hallucination_join(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group means, point-biserial r and AUROC per statistic (see LIMITS_NOTE)."""
    join: dict[str, dict[str, Any]] = {}
    for statistic in STATISTICS:
        pairs = [
            (bool(record["has_hallucination"]), record["statistics"][statistic])
            for record in records
            if record["statistics"][statistic] is not None
        ]
        labels = [label for label, _ in pairs]
        values = [value for _, value in pairs]
        halluc = [value for label, value in pairs if label]
        clean = [value for label, value in pairs if not label]
        mean_halluc = _mean(halluc)
        mean_clean = _mean(clean)
        join[statistic] = {
            "n_hallucinating": len(halluc),
            "n_clean": len(clean),
            "mean_hallucinating": _r(mean_halluc),
            "mean_clean": _r(mean_clean),
            "delta_halluc_minus_clean": _r(
                mean_halluc - mean_clean
                if mean_halluc is not None and mean_clean is not None
                else None
            ),
            "point_biserial_r": _r(_point_biserial(labels, values)),
            "auroc_hallucinating_vs_clean": _r(_auroc(labels, values)),
        }
    return join


def main() -> int:
    args = parse_args()

    lenses, provenance = load_lens_set(args.lens_dir)
    if "text" not in lenses:
        raise ValueError(
            f"mask 'text' not in {args.lens_dir} (found {sorted(lenses)})"
        )
    lens = lenses["text"]
    if not lens.source_layers:
        raise ValueError(f"lens 'text' in {args.lens_dir} has no fitted layers")
    layers = list(lens.source_layers)

    model = load_model(args)
    invalid = [layer for layer in layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"fitted layers {invalid} are outside the {args.backend} model "
            f"(n_layers={model.n_layers}); the lens dir does not match this backend"
        )
    mid_layer = layers[len(layers) // 2]

    category_ids, per_file_categories = x14_workspace_probe.load_coco_grounding(
        args.annotations
    )
    patterns = x14_workspace_probe._category_patterns(category_ids)
    images = select_images(list_coco_images(args.images_dir), args.n_images, seed=args.seed)
    prompt = prompt_text(args.question)

    config = model.hf_model.config
    report: dict[str, Any] = {
        "experiment": "x17_attention_mirror",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "attn_implementation_requested": args.attn_impl,
        "attn_implementation_effective": config._attn_implementation,
        "lens_dir": args.lens_dir,
        "mask": "text",
        "fitted_layers": layers,
        "mid_layer": mid_layer,
        "n_prompts": int(lens.n_prompts),
        "n_layers": int(model.n_layers),
        "d_model": int(model.d_model),
        "n_heads_llm": int(config.get_text_config().num_attention_heads),
        "n_vision_layers": int(config.vision_config.num_hidden_layers),
        "n_heads_vision": int(config.vision_config.num_attention_heads),
        "vision_fingerprint": model.vision_fingerprint(),
        "lens_provenance": provenance or None,
        "images_dir": args.images_dir,
        "annotations": args.annotations,
        "n_images": len(images),
        "seed": args.seed,
        "question": args.question,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_hash": prompt_template_hash(),
        "query_mode": args.query_mode,
        "query_mode_note": QUERY_MODE_NOTE[args.query_mode],
        "clip_mid_layer": args.clip_mid_layer,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "attention_share_note": ATTENTION_SHARE_NOTE,
        "patch_alignment_note": ALIGNMENT_NOTE,
        "matcher_note": x14_workspace_probe.MATCHER_NOTE,
        "matcher_source": "x14_workspace_probe.classify_generated_words",
        "interpretation_limits": LIMITS_NOTE,
    }
    print(
        f"== X17 attention mirror: {len(images)} images, layers={layers} "
        f"(mid L{mid_layer}), mode={args.query_mode}, attn={args.attn_impl} =="
    )

    records: list[dict[str, Any]] = []
    n_skipped = 0
    for index, path in enumerate(images):
        present = per_file_categories.get(path.name)
        if not present:
            print(
                f"  [{index + 1}/{len(images)}] {path.name}: no annotations entry - skipped",
                flush=True,
            )
            n_skipped += 1
            continue

        batch = model.encode_mm(prompt, path, max_length=args.max_seq_len)
        llm = llm_image_attention(model, batch, layers, args.query_mode)
        clip = clip_cls_patch_attention(model, batch, args.clip_mid_layer)
        if clip["final"]["patch_vector"].shape[0] != batch.n_image_tokens:
            raise RuntimeError(
                f"CLIP patch vector ({clip['final']['patch_vector'].shape[0]}) does not "
                f"align with the prompt's {batch.n_image_tokens} image tokens "
                f"(select strategy {model.vision_feature_select_strategy!r}); "
                "see patch_alignment_note"
            )

        caption_out = generate_with_edits(
            model, None, batch, max_new_tokens=args.max_new_tokens
        )
        new_token_ids = torch.tensor(caption_out["token_ids"], dtype=torch.long)
        caption, instances = x14_workspace_probe.classify_generated_words(
            model.tokenizer, new_token_ids, patterns, present
        )
        hallucinated = sorted({word.category for word in instances if not word.grounded})
        grounded = sorted({word.category for word in instances if word.grounded})

        clip_final = clip["final"]["patch_vector"].numpy()
        spearman_rho = {
            str(layer): _spearman(clip_final, llm[layer]["patch_vector"].numpy())
            for layer in layers
        }
        spearman_rho_clip_mid = (
            {
                str(layer): _spearman(
                    clip["mid"]["patch_vector"].numpy(), llm[layer]["patch_vector"].numpy()
                )
                for layer in layers
            }
            if "mid" in clip
            else None
        )

        shares = [llm[layer]["share"] for layer in layers]
        rhos = [rho for rho in spearman_rho.values() if rho is not None]
        statistics = {
            "share_mid_layer": llm[mid_layer]["share"],
            "share_mean_over_layers": _mean(shares),
            "rho_mid_layer": spearman_rho[str(mid_layer)],
            "rho_mean_over_layers": _mean(rhos),
        }

        records.append(
            {
                "image": path.name,
                "seq_len": batch.seq_len,
                "n_image_tokens": batch.n_image_tokens,
                "caption": caption,
                "caption_n_tokens": len(caption_out["token_ids"]),
                "mentioned": sorted({word.category for word in instances}),
                "hallucinated": hallucinated,
                "grounded": grounded,
                "has_hallucination": bool(hallucinated),
                "llm_attention": {
                    str(layer): _layer_record(llm[layer]) for layer in layers
                },
                "clip_attention": {
                    name: _clip_record(record) for name, record in clip.items()
                },
                "spearman_rho": {str(layer): _r(rho) for layer, rho in spearman_rho.items()},
                "spearman_rho_clip_mid": (
                    {str(layer): _r(rho) for layer, rho in spearman_rho_clip_mid.items()}
                    if spearman_rho_clip_mid is not None
                    else None
                ),
                "statistics": {name: _r(value) for name, value in statistics.items()},
            }
        )
        print(
            f"  [{index + 1}/{len(images)}] {path.name}: seq={batch.seq_len} "
            f"img={batch.n_image_tokens} gen={len(caption_out['token_ids'])} "
            f"matched={len(instances)} ({len(grounded)}g/{len(hallucinated)}h) "
            f"rho={_cell(statistics['rho_mid_layer'], 5, 3)} {caption[:60]!r}",
            flush=True,
        )

    per_layer: dict[str, dict[str, Any]] = {}
    for layer in layers:
        rows = [record["llm_attention"][str(layer)] for record in records]
        rhos = [
            record["spearman_rho"][str(layer)]
            for record in records
            if record["spearman_rho"][str(layer)] is not None
        ]
        hall = [record for record in records if record["has_hallucination"]]
        clean = [record for record in records if not record["has_hallucination"]]
        per_layer[str(layer)] = {
            "share_mean": _r(_mean([row["share"] for row in rows])),
            "patch_mean_mean": _r(_mean([row["patch_mean"] for row in rows])),
            "patch_max_mean": _r(_mean([row["patch_max"] for row in rows])),
            "rho_mean": _r(_mean([float(rho) for rho in rhos])),
            "n_rho": len(rhos),
            "share_mean_hallucinating": _r(
                _mean([record["llm_attention"][str(layer)]["share"] for record in hall])
            ),
            "share_mean_clean": _r(
                _mean([record["llm_attention"][str(layer)]["share"] for record in clean])
            ),
        }

    join = hallucination_join(records)
    rhos_by_layer = {
        layer: per_layer[str(layer)]["rho_mean"]
        for layer in layers
        if per_layer[str(layer)]["rho_mean"] is not None
    }
    deltas_by_statistic = {
        statistic: entry["delta_halluc_minus_clean"]
        for statistic, entry in join.items()
        if entry["delta_halluc_minus_clean"] is not None
    }
    report["images"] = records
    report["per_layer"] = per_layer
    report["hallucination_join"] = join
    report["summary"] = {
        "n_images_analyzed": len(records),
        "n_images_skipped_no_annotations": n_skipped,
        "n_with_hallucination": sum(1 for record in records if record["has_hallucination"]),
        "n_hallucinated_mentions": sum(len(record["hallucinated"]) for record in records),
        "n_grounded_mentions": sum(len(record["grounded"]) for record in records),
        "best_abs_rho_layer": (
            max(rhos_by_layer, key=lambda layer: abs(rhos_by_layer[layer]))
            if rhos_by_layer
            else None
        ),
        "largest_group_delta_statistic": (
            max(deltas_by_statistic, key=lambda statistic: abs(deltas_by_statistic[statistic]))
            if deltas_by_statistic
            else None
        ),
    }
    report["digest"] = {
        "note": "see per_layer + hallucination_join; every number is CORRELATIONAL",
        "interpretation_limits": LIMITS_NOTE,
    }

    summary = report["summary"]
    print(
        f"\n== digest: {summary['n_images_analyzed']} images, "
        f"{summary['n_with_hallucination']} with a hallucination, "
        f"{summary['n_hallucinated_mentions']} hallucinated / "
        f"{summary['n_grounded_mentions']} grounded mentions =="
    )
    print(f"{'layer':>6}{'share':>9}{'p_mean':>9}{'p_max':>9}{'rho':>9}{'n_rho':>6}"
          f" | {'sh_hal':>9}{'sh_clean':>9}")
    for layer in layers:
        row = per_layer[str(layer)]
        print(
            f"{layer:>6}"
            f"{_cell(row['share_mean'], 9)}"
            f"{_cell(row['patch_mean_mean'], 9)}"
            f"{_cell(row['patch_max_mean'], 9)}"
            f"{_cell(row['rho_mean'], 9)}"
            f"{row['n_rho']:>6} | "
            f"{_cell(row['share_mean_hallucinating'], 9)}"
            f"{_cell(row['share_mean_clean'], 9)}"
        )
    print(f"\n== hallucination join (positive class = hallucinating image; CORRELATIONAL) ==")
    print(f"{'statistic':<26}{'n_hal':>6}{'n_clean':>8}{'mean_hal':>10}{'mean_clean':>11}"
          f"{'delta':>9}{'r_pb':>8}{'auroc':>8}")
    for statistic in STATISTICS:
        entry = join[statistic]
        print(
            f"{statistic:<26}{entry['n_hallucinating']:>6}{entry['n_clean']:>8}"
            f"{_cell(entry['mean_hallucinating'], 10)}"
            f"{_cell(entry['mean_clean'], 11)}"
            f"{_cell(entry['delta_halluc_minus_clean'], 9)}"
            f"{_cell(entry['point_biserial_r'], 8)}"
            f"{_cell(entry['auroc_hallucinating_vs_clean'], 8)}"
        )
    print(f"\nlimits: {LIMITS_NOTE}")

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json} ({len(records)} images, {n_skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
