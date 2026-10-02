#!/usr/bin/env python
"""End-to-end CPU smoke harness for the validation campaign's analysis scripts.

Builds a miniature copy of the campaign fixtures (manifests, PNGs, corpus split, X3-norms
stub), fits tiny lenses with the production fit CLI, merges them, then runs every analysis
script as a subprocess on its ``--backend tiny`` model and validates the JSON report it
writes. Purpose: catch "never-executed script" bugs before the real GPU campaign runs, not
to test the library (the library has its own pytest suite).

Usage (from the repo root):

    PYTHONPATH=src ~/miniforge3/envs/vlm-lens/bin/python \
        results/validation_2026-10-01/code/smoke_analysis.py

Re-running is idempotent: lens fits resume from their checkpoints and reports are
overwritten. Pass ``--fresh`` to wipe the fixture directory first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from vlm_lens.artifacts import load_lens_set  # noqa: E402
from vlm_lens.data.captions import (  # noqa: E402
    PROMPT_TEMPLATE,
    prompt_template_hash,
    prompt_text,
)
from vlm_lens.data.manifest import FitSample, write_manifest  # noqa: E402

SMOKE_VERSION = 1
DEFAULT_OUT = Path("/tmp/smoke_analysis")
PYTHON = sys.executable
#: Must match ``fit_llava.py --backend tiny`` defaults (TinyLlavaConfig: d_model=32,
#: n_layers=4, image_size=336, 576 image tokens, final layer 3).
IMAGE_SIZE = 336
QUESTION_A = "Describe this image."
QUESTION_B = "What is happening in this image?"
FIT_MASKS = "text,image,all"
#: Tiny-model source layers (must stay below the final/target layer 3).
LAYERS = "0,1,2"
EDIT_LAYERS = "1,2"
X4_SKIP_FIRSTS = (1, 8, 16, 32)
N_FIT_PER_HALF = 3
N_HELDOUT = 6
N_TEXT_FIT = 2
N_TEXT_HELDOUT = 2
TEXT_N_WORDS = 64
X7_N_SAMPLES = 2
#: add(3) + ablate(2) + swap(2) default alpha grid in x7_x9_interventions.py.
X9_EDITS_PER_PAIR = 7

ENV = {
    **os.environ,
    "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
}


class StepError(RuntimeError):
    """A subprocess step failed; carries the captured output for the report."""

    def __init__(self, label: str, command: str, output: str):
        super().__init__(f"{label} failed: {command}")
        self.output = output


def run_cmd(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    printable = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"  $ {printable}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
    elapsed = time.time() - started
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, flush=True)
        raise StepError(label, printable, proc.stdout + proc.stderr)
    tail = [line for line in proc.stdout.splitlines() if line.strip()]
    if tail:
        print(f"    {tail[-1]}", flush=True)
    print(f"    [{label}: rc=0 in {elapsed:.1f}s]", flush=True)
    return proc


# ---------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------


def fixture_signature() -> str:
    """Hash of the fixture plan; a change wipes the cached fixture directory."""
    spec = {
        "smoke_version": SMOKE_VERSION,
        "questions": [QUESTION_A, QUESTION_B],
        "image_size": IMAGE_SIZE,
        "counts": [N_FIT_PER_HALF, N_HELDOUT, N_TEXT_FIT, N_TEXT_HELDOUT, X7_N_SAMPLES],
        "text_n_words": TEXT_N_WORDS,
        "masks": FIT_MASKS,
        "x4_skip_firsts": list(X4_SKIP_FIRSTS),
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()


def _image_names() -> list[str]:
    return [f"img{index:02d}.png" for index in range(2 * N_FIT_PER_HALF + N_HELDOUT)]


def _caption_sample(image_dir: Path, name: str, question: str, caption: str, tag: str) -> FitSample:
    return FitSample(
        sample_id=f"{name}::prompt+caption::{tag}",
        text=prompt_text(question, caption),
        images=(str(image_dir / name),),
        meta={
            "corpus": "coco_val2014_smoke",
            "question": question,
            "caption": caption,
            "variant": "prompt+caption",
            "image": name,
            "generation": {"do_sample": False, "max_new_tokens": 0, "model": "tiny-smoke"},
        },
    )


def _text_sample(tag: str, index: int) -> FitSample:
    text = " ".join(f"word{word % 53}" for word in range(TEXT_N_WORDS))
    return FitSample(
        sample_id=f"{tag}-{index}",
        text=f"{text} sample {index}",
        images=(),
        meta={"corpus": "wikitext_smoke", "n_chars": len(text)},
    )


def _caption_meta(images_dir: Path, names: list[str]) -> dict:
    return {
        "corpus": "coco_val2014_smoke",
        "images_dir": str(images_dir),
        "image_list_given": True,
        "image_names": list(names),
        "questions_per_image": 1,
        "seed": 1234,
        "prompt_mode": "prompt+caption",
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": prompt_template_hash(),
        "max_new_tokens": 0,
        "max_seq_len": 1536,
    }


def build_fixture(out: Path, *, fresh: bool) -> dict[str, Path]:
    """Write images, manifests, split sidecar, norms stub; return all fixture paths."""
    marker = out / "smoke_marker.json"
    signature = fixture_signature()
    if fresh and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        if marker.exists():
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old.get("signature") != signature:
                print(f"[smoke] fixture spec changed; wiping {out}")
                shutil.rmtree(out)
        else:
            raise StepError(
                "fixture",
                str(out),
                f"{out} is not a smoke_analysis fixture directory; pass --fresh to overwrite",
            )

    images_dir = out / "images"
    manifests = out / "manifests"
    reports = out / "reports"
    for directory in (images_dir, manifests, reports):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1234)
    names = _image_names()
    for name in names:
        array = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(array).save(images_dir / name)

    half_a_names = names[:N_FIT_PER_HALF]
    half_b_names = names[N_FIT_PER_HALF : 2 * N_FIT_PER_HALF]
    heldout_names = names[2 * N_FIT_PER_HALF :]

    half_a = [
        _caption_sample(images_dir, name, QUESTION_A, f"a photo of object {index} in a room", "fit-a")
        for index, name in enumerate(half_a_names)
    ]
    half_b = [
        _caption_sample(images_dir, name, QUESTION_B, f"a photo of object {index} in a room", "fit-b")
        for index, name in enumerate(half_b_names)
    ]
    heldout_questions = [
        QUESTION_A if index % 2 == 0 else QUESTION_B for index in range(len(heldout_names))
    ]
    heldout = [
        _caption_sample(images_dir, name, question, f"a photo of object {index} in a room", "heldout")
        for index, (name, question) in enumerate(zip(heldout_names, heldout_questions))
    ]
    text_fit = [_text_sample("text-fit", index) for index in range(N_TEXT_FIT)]
    text_heldout = [_text_sample("text-heldout", index) for index in range(N_TEXT_HELDOUT)]

    paths = {
        "out": out,
        "images": images_dir,
        "reports": reports,
        "half_a_manifest": manifests / "half-a.jsonl",
        "half_b_manifest": manifests / "half-b.jsonl",
        "heldout_manifest": manifests / "heldout.jsonl",
        "text_fit_manifest": manifests / "text-fit.jsonl",
        "text_heldout_manifest": manifests / "text-heldout.jsonl",
        "split_json": out / "corpus-split.json",
        "norms_json": out / "x3-norms.json",
        "lens_s2_half_a": out / "lenses" / "s2-half-a",
        "lens_s2_half_b": out / "lenses" / "s2-half-b",
        "lens_s2_merged": out / "lenses" / "s2-merged",
        "lens_s1_text": out / "lenses" / "s1-text",
        "lens_x1_all": out / "lenses" / "x1-all",
        "lens_x1_text": out / "lenses" / "x1-text",
    }
    for skip_first in X4_SKIP_FIRSTS:
        paths[f"lens_x4_sf{skip_first}"] = out / "lenses" / f"x4-sf{skip_first}"

    write_manifest(paths["half_a_manifest"], half_a, meta=_caption_meta(images_dir, half_a_names))
    write_manifest(paths["half_b_manifest"], half_b, meta=_caption_meta(images_dir, half_b_names))
    write_manifest(paths["heldout_manifest"], heldout, meta=_caption_meta(images_dir, heldout_names))
    write_manifest(
        paths["text_fit_manifest"], text_fit, meta={"corpus": "wikitext_smoke", "source": "synthetic"}
    )
    write_manifest(
        paths["text_heldout_manifest"],
        text_heldout,
        meta={"corpus": "wikitext_smoke", "source": "synthetic"},
    )

    split = {
        "images_dir": str(images_dir),
        "seed": 1234,
        "fit_images": half_a_names + half_b_names,
        "held_out_images": heldout_names,
        "half_a_questions": [QUESTION_A],
        "half_b_questions": [QUESTION_B],
        "fit_manifest": str(paths["half_a_manifest"]),
        "heldout_manifest": str(paths["heldout_manifest"]),
    }
    paths["split_json"].write_text(json.dumps(split, indent=2), encoding="utf-8")

    norms = {
        "manifest": str(paths["half_a_manifest"]),
        "n_samples": N_HELDOUT,
        "layers": {
            str(layer): {
                group: {"mean_norm": base + 0.25 * layer}
                for base, group in (
                    (0.75, "bos"),
                    (0.9, "pos_1_16"),
                    (1.1, "text_pre"),
                    (1.3, "patch_1_16"),
                    (1.5, "patch_17_end"),
                    (2.0, "text_post"),
                )
            }
            for layer in range(4)
        },
    }
    paths["norms_json"].write_text(json.dumps(norms, indent=2), encoding="utf-8")

    partial = {
        "manifest": str(paths["half_a_manifest"]),
        "n_samples": N_HELDOUT,
        "layers": {layer: groups for layer, groups in norms["layers"].items() if layer == "1"},
    }
    paths["norms_partial_json"] = out / "x3-norms-partial.json"
    paths["norms_partial_json"].write_text(json.dumps(partial, indent=2), encoding="utf-8")

    marker.write_text(
        json.dumps(
            {
                "smoke_version": SMOKE_VERSION,
                "signature": signature,
                "fixture_dir": str(out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def _fit_jobs(paths: dict[str, Path]) -> list[dict]:
    return [
        {"name": "s2-half-a", "manifest": paths["half_a_manifest"], "masks": FIT_MASKS, "skip_first": 1, "target_mask": None, "out": paths["lens_s2_half_a"]},
        {"name": "s2-half-b", "manifest": paths["half_b_manifest"], "masks": FIT_MASKS, "skip_first": 1, "target_mask": None, "out": paths["lens_s2_half_b"]},
        {"name": "s1-text", "manifest": paths["text_fit_manifest"], "masks": "all", "skip_first": 16, "target_mask": None, "out": paths["lens_s1_text"]},
        {"name": "x1-all", "manifest": paths["half_a_manifest"], "masks": FIT_MASKS, "skip_first": 1, "target_mask": "all", "out": paths["lens_x1_all"]},
        {"name": "x1-text", "manifest": paths["half_a_manifest"], "masks": FIT_MASKS, "skip_first": 1, "target_mask": "text", "out": paths["lens_x1_text"]},
    ] + [
        {"name": f"x4-sf{skip_first}", "manifest": paths["half_a_manifest"], "masks": FIT_MASKS, "skip_first": skip_first, "target_mask": None, "out": paths[f"lens_x4_sf{skip_first}"]}
        for skip_first in X4_SKIP_FIRSTS
    ]


def fit_lenses(paths: dict[str, Path]) -> None:
    for job in _fit_jobs(paths):
        cmd = [
            PYTHON, str(REPO / "scripts" / "fit_llava.py"),
            "--backend", "tiny", "--manifest", str(job["manifest"]),
            "--layers", "all", "--masks", job["masks"], "--dim-batch", "8",
            "--skip-first", str(job["skip_first"]), "--checkpoint-every", "1",
            "--out", str(job["out"]), "--notes", "smoke",
        ]
        if job["target_mask"]:
            cmd += ["--target-mask", job["target_mask"]]
        run_cmd(cmd, f"fit {job['name']}")
    run_cmd(
        [
            PYTHON, str(REPO / "scripts" / "fit_llava.py"),
            "--merge", str(paths["lens_s2_half_a"]), str(paths["lens_s2_half_b"]),
            "--out", str(paths["lens_s2_merged"]), "--notes", "smoke merge",
        ],
        "merge s2 halves",
    )


def check_merge(paths: dict[str, Path]) -> list[str]:
    """Merged lens set must load with summed n_prompts and the shard-weighted mean."""
    problems: list[str] = []
    half_a, _ = load_lens_set(paths["lens_s2_half_a"] / "artifacts")
    half_b, _ = load_lens_set(paths["lens_s2_half_b"] / "artifacts")
    merged, _ = load_lens_set(paths["lens_s2_merged"] / "artifacts")
    if set(merged) != {"text", "image", "all"}:
        problems.append(f"merged masks={sorted(merged)}")
    for mask in sorted(set(merged) & set(half_a) & set(half_b)):
        lens_a, lens_b, lens_m = half_a[mask], half_b[mask], merged[mask]
        total = int(lens_a.n_prompts) + int(lens_b.n_prompts)
        if int(lens_m.n_prompts) != total:
            problems.append(f"{mask}: n_prompts={lens_m.n_prompts} != {total}")
        if sorted(lens_m.source_layers) != sorted(lens_a.source_layers):
            problems.append(
                f"{mask}: source_layers={sorted(lens_m.source_layers)} != {sorted(lens_a.source_layers)}"
            )
        for layer in sorted(set(lens_m.jacobians) & set(lens_a.jacobians) & set(lens_b.jacobians)):
            expected = (
                int(lens_a.n_prompts) * lens_a.jacobians[layer].float()
                + int(lens_b.n_prompts) * lens_b.jacobians[layer].float()
            ) / total
            actual = lens_m.jacobians[layer].float()
            relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
            if relative > 5e-2:  # storage-dtype rounding in merged artifacts (observed ~3e-4)
                problems.append(f"{mask} L{layer}: weighted-mean rel err {relative:.3f}")
    return problems


# ---------------------------------------------------------------------------------------
# report validation
# ---------------------------------------------------------------------------------------


def _require(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def check_s2_report(data: dict, *, merged_n_prompts: int, half_n_prompts: int) -> list[str]:
    problems: list[str] = []
    for key in (
        "heldout_manifest", "n_heldout_samples", "skip_first", "tags",
        "mask_composition", "main", "halves", "transfer",
    ):
        _require(key in data, f"missing top-level key {key!r}", problems)
    composition = data.get("mask_composition") or {}
    _require(len(composition) > 1 and "seq_len" in composition, f"mask_composition={composition}", problems)

    main = data.get("main") or {}
    _require(main.get("n_prompts") == merged_n_prompts, f"main.n_prompts={main.get('n_prompts')} != {merged_n_prompts}", problems)
    _require(main.get("layers") == [0, 1, 2], f"main.layers={main.get('layers')}", problems)
    for mode in ("default", "include_placeholders"):
        rows = ((main.get("rows") or {}).get(mode)) or []
        _require(len(rows) >= 4, f"main.rows.{mode}: {len(rows)} rows", problems)
        _require(
            all(isinstance(row.get("layer"), int) and "mean_kl" in row and "mean_rank_true" in row for row in rows),
            f"main.rows.{mode} malformed",
            problems,
        )

    halves = data.get("halves") or {}
    n_samples = halves.get("n_samples") or {}
    _require(n_samples.get("half_a", 0) > 0 and n_samples.get("half_b", 0) > 0, f"halves.n_samples={n_samples}", problems)
    for key in ("lens_a_on_half_a", "lens_a_on_half_b", "lens_b_on_half_a", "lens_b_on_half_b"):
        entry = halves.get(key) or {}
        _require(entry.get("n_prompts") == half_n_prompts, f"halves[{key}].n_prompts={entry.get('n_prompts')} != {half_n_prompts}", problems)
        _require(len(entry.get("rows") or []) >= 1, f"halves[{key}].rows empty", problems)

    transfer = data.get("transfer") or {}
    for key in ("caption_lens_on_wikitext", "text_lens_on_captions"):
        _require(len(((transfer.get(key) or {}).get("rows")) or []) >= 1, f"transfer[{key}].rows empty", problems)
    return problems


def check_x1_report(data: dict) -> list[str]:
    problems: list[str] = []
    for key in ("dir_all", "dir_text", "n_prompts", "provenance_target_mask", "sample_geometry", "bit_check", "lens_comparison"):
        _require(key in data, f"missing top-level key {key!r}", problems)
    provenance = data.get("provenance_target_mask") or {}
    _require((provenance.get("all") or {}).get("text") == "all", f"provenance_target_mask.all={provenance.get('all')}", problems)
    _require((provenance.get("text") or {}).get("text") == "text", f"provenance_target_mask.text={provenance.get('text')}", problems)
    positions = ((data.get("sample_geometry") or {}).get("mask_positions")) or {}
    for mask in ("text", "image", "all"):
        _require(positions.get(mask, 0) > 0, f"sample_geometry.mask_positions[{mask}]={positions.get(mask)}", problems)
    _require(
        (data.get("sample_geometry") or {}).get("pre_image_text_positions", 0) >= 1,
        "sample_geometry.pre_image_text_positions < 1: no pre-image text in the fixture "
        "prompt, so the causally-disjoint subset check would be vacuous",
        problems,
    )
    bit = data.get("bit_check") or {}
    per_mask = bit.get("per_mask") or {}
    _require(set(per_mask) == {"text", "image", "all"}, f"bit_check.per_mask keys={sorted(per_mask)}", problems)
    _require(len(per_mask.get("text") or {}) == 3, f"bit_check.per_mask.text layers={len(per_mask.get('text') or {})}", problems)
    _require(bit.get("text_rows_bit_identical") is True, "X1 verdict: text-block rows are NOT bit-identical", problems)
    _require(bool((data.get("lens_comparison") or {}).get("text")), "lens_comparison.text missing/empty", problems)
    return problems


def check_x4_report(data: dict, skip_firsts: tuple[int, ...]) -> list[str]:
    problems: list[str] = []
    _require(data.get("n_heldout") == N_HELDOUT, f"n_heldout={data.get('n_heldout')}", problems)
    for skip_first in skip_firsts:
        key = str(skip_first)
        entry = data.get(key) or {}
        _require(entry.get("n_prompts") == N_FIT_PER_HALF, f"x4[{key}].n_prompts={entry.get('n_prompts')}", problems)
        rows = entry.get("rows") or []
        _require(len(rows) >= 3, f"x4[{key}].rows={len(rows)}", problems)
        _require(all("mean_rank_true" in row and "mean_kl" in row for row in rows), f"x4[{key}] rows malformed", problems)
    summary = data.get("summary_l31_text") or {}
    _require(set(summary) == {str(skip_first) for skip_first in skip_firsts}, f"summary_l31_text keys={sorted(summary)}", problems)
    for key, row in summary.items():
        _require("mean_rank_true" in row and "mean_kl" in row, f"summary_l31_text[{key}] malformed", problems)
    return problems


def check_x7_report(data: dict, *, edit_layers: tuple[int, ...], merged_n_prompts: int) -> list[str]:
    problems: list[str] = []
    for key in (
        "lens_dir", "n_prompts", "n_samples", "edit_layers", "residual_norms", "fallback_norm",
        "x7_conditioning", "x9_usable_pairs", "x9_baselines", "x9_results", "x9_summary",
    ):
        _require(key in data, f"missing top-level key {key!r}", problems)
    _require(data.get("n_prompts") == merged_n_prompts, f"n_prompts={data.get('n_prompts')}", problems)
    _require(data.get("n_samples") == X7_N_SAMPLES, f"n_samples={data.get('n_samples')}", problems)
    _require(data.get("edit_layers") == list(edit_layers), f"edit_layers={data.get('edit_layers')}", problems)

    conditioning = data.get("x7_conditioning") or {}
    for layer in edit_layers:
        per_pair = conditioning.get(str(layer)) or {}
        _require(len(per_pair) == 10, f"x7_conditioning[{layer}] pairs={len(per_pair)}", problems)
        _require(
            all(entry.get("cond", 0) > 0 and -1.001 <= entry.get("cosine", 99) <= 1.001 for entry in per_pair.values()),
            f"x7_conditioning[{layer}] malformed entries",
            problems,
        )
    norms = data.get("residual_norms") or {}
    for layer in edit_layers:
        value = norms.get(str(layer))
        _require(isinstance(value, (int, float)) and value > 0, f"residual_norms[{layer}]={value} (X3 norms JSON not read)", problems)

    baselines = data.get("x9_baselines") or {}
    _require(len(baselines) == X7_N_SAMPLES, f"x9_baselines={len(baselines)}", problems)
    for key, entry in baselines.items():
        _require(isinstance(entry.get("text"), str), f"baselines[{key}].text", problems)
        _require(isinstance(entry.get("token_ids"), list) and len(entry["token_ids"]) >= 1, f"baselines[{key}].token_ids", problems)

    usable = data.get("x9_usable_pairs") or []
    _require(len(usable) >= 1, "x9_usable_pairs empty: X9 generated nothing", problems)
    results = data.get("x9_results") or []
    expected = X7_N_SAMPLES * len(edit_layers) * len(usable) * X9_EDITS_PER_PAIR
    _require(len(results) == expected, f"x9_results={len(results)} != {expected}", problems)
    for row in results:
        _require(isinstance(row.get("changed"), bool), "x9 row changed not bool", problems)
        _require(isinstance(row.get("n_edit_forwards"), int) and row["n_edit_forwards"] >= 1, f"n_edit_forwards={row.get('n_edit_forwards')}", problems)
    summary = data.get("x9_summary") or {}
    _require(summary.get("n_generations") == len(results), f"x9_summary.n_generations={summary.get('n_generations')} != {len(results)}", problems)
    return problems


def check_x7_fallback(
    data: dict,
    *,
    edit_layers: tuple[int, ...],
    missing_layer: int,
    n_samples: int,
    n_prompts: int,
) -> list[str]:
    """X7/X9 with a census lacking one edit layer: the norm falls back, the run completes."""
    problems: list[str] = []
    _require(data.get("n_prompts") == n_prompts, f"n_prompts={data.get('n_prompts')}", problems)
    _require(data.get("n_samples") == n_samples, f"n_samples={data.get('n_samples')}", problems)
    norms = data.get("residual_norms") or {}
    for layer in edit_layers:
        value = norms.get(str(layer))
        if layer == missing_layer:
            _require(value is None, f"residual_norms[{layer}]={value}, expected None", problems)
        else:
            _require(
                isinstance(value, (int, float)) and value > 0,
                f"residual_norms[{layer}]={value}",
                problems,
            )
    fallback = data.get("fallback_norm")
    _require(float(fallback or 0) > 0, f"fallback_norm={fallback}", problems)
    usable = data.get("x9_usable_pairs") or []
    _require(len(usable) >= 1, "x9_usable_pairs empty: the fallback run generated nothing", problems)
    results = data.get("x9_results") or []
    expected = n_samples * len(edit_layers) * len(usable) * X9_EDITS_PER_PAIR
    _require(len(results) == expected, f"x9_results={len(results)} != {expected}", problems)
    summary = data.get("x9_summary") or {}
    _require(
        summary.get("n_generations") == len(results),
        f"x9_summary.n_generations={summary.get('n_generations')} != {len(results)}",
        problems,
    )
    return problems


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="fixture directory (default: %(default)s)")
    parser.add_argument("--fresh", action="store_true", help="wipe --out before building fixtures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        suffix = "" if ok or not detail else f" -- {detail}"
        print(f"{'PASS' if ok else 'FAIL'}: {name}{suffix}", flush=True)

    try:
        paths = build_fixture(out, fresh=args.fresh)
    except Exception as exc:  # noqa: BLE001 - harness reports any fixture failure
        print(f"FAIL: fixture build -- {exc}", flush=True)
        return 1
    record("fixture build (images, manifests, split, x3-norms)", True)
    print(f"[smoke] fixtures at {out}", flush=True)

    try:
        fit_lenses(paths)
    except Exception as exc:  # noqa: BLE001
        record("lens fits", False, str(exc))
        return 1

    try:
        merge_problems = check_merge(paths)
    except Exception as exc:  # noqa: BLE001
        merge_problems = [f"{type(exc).__name__}: {exc}"]
    record(
        "fit_llava --merge: loads, n_prompts summed, shard-weighted mean",
        not merge_problems,
        "; ".join(merge_problems),
    )

    merged_n_prompts = N_FIT_PER_HALF * 2
    script_checks: list[tuple[str, list[str], object]] = [
        (
            "s2_eval.py",
            [
                "--main-lens-dir", str(paths["lens_s2_merged"] / "artifacts"),
                "--heldout-manifest", str(paths["heldout_manifest"]),
                "--half-lens-a", str(paths["lens_s2_half_a"] / "artifacts"),
                "--half-lens-b", str(paths["lens_s2_half_b"] / "artifacts"),
                "--split-json", str(paths["split_json"]),
                "--text-lens-dir", str(paths["lens_s1_text"] / "artifacts"),
                "--text-heldout", str(paths["text_heldout_manifest"]),
                "--backend", "tiny",
            ],
            lambda data: check_s2_report(data, merged_n_prompts=merged_n_prompts, half_n_prompts=N_FIT_PER_HALF),
        ),
        (
            "x1_compare.py",
            [
                "--dir-all", str(paths["lens_x1_all"] / "artifacts"),
                "--dir-text", str(paths["lens_x1_text"] / "artifacts"),
                "--manifest", str(paths["half_a_manifest"]),
                "--sample-index", "0", "--layers", LAYERS, "--dtype", "float32",
                "--backend", "tiny",
            ],
            check_x1_report,
        ),
        (
            "x4_eval.py",
            [
                "--variants",
                *[f"{skip_first}={paths[f'lens_x4_sf{skip_first}'] / 'artifacts'}" for skip_first in X4_SKIP_FIRSTS],
                "--heldout-manifest", str(paths["heldout_manifest"]),
                "--backend", "tiny",
            ],
            lambda data: check_x4_report(data, X4_SKIP_FIRSTS),
        ),
        (
            "x7_x9_interventions.py",
            [
                "--lens-dir", str(paths["lens_s2_merged"] / "artifacts"),
                "--manifest", str(paths["heldout_manifest"]),
                "--n-samples", str(X7_N_SAMPLES), "--layers", LAYERS, "--edit-layers", EDIT_LAYERS,
                "--norms-json", str(paths["norms_json"]),
                "--backend", "tiny",
            ],
            lambda data: check_x7_report(data, edit_layers=tuple(int(part) for part in EDIT_LAYERS.split(",")), merged_n_prompts=merged_n_prompts),
        ),
    ]

    for script, extra, checker in script_checks:
        report_path = paths["reports"] / script.replace(".py", ".json")
        try:
            run_cmd([PYTHON, str(CODE / script), *extra, "--json", str(report_path)], script)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            problems = checker(report)  # type: ignore[operator]
        except StepError as exc:
            record(f"{script} (tiny backend)", False, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            record(f"{script} (tiny backend)", False, f"{type(exc).__name__}: {exc}")
            continue
        record(f"{script} (tiny backend)", not problems, "; ".join(problems))

    # The production X3 census holds layers 0/16/31 while X9 edits at 16/24, so a missing
    # edit-layer norm is a live path on the real run: exercise the fallback with a census
    # that lacks the top edit layer.
    fallback_edit_layers = tuple(int(part) for part in EDIT_LAYERS.split(","))
    fallback_report = paths["reports"] / "x7_x9_fallback.json"
    fallback_label = "x7_x9_interventions.py fallback norm (census missing an edit layer)"
    try:
        run_cmd(
            [
                PYTHON, str(CODE / "x7_x9_interventions.py"),
                "--lens-dir", str(paths["lens_s2_merged"] / "artifacts"),
                "--manifest", str(paths["heldout_manifest"]),
                "--n-samples", "1", "--layers", LAYERS, "--edit-layers", EDIT_LAYERS,
                "--norms-json", str(paths["norms_partial_json"]),
                "--max-new-tokens", "8", "--backend", "tiny",
                "--json", str(fallback_report),
            ],
            "x7_x9 fallback norm",
        )
        report = json.loads(fallback_report.read_text(encoding="utf-8"))
        problems = check_x7_fallback(
            report,
            edit_layers=fallback_edit_layers,
            missing_layer=max(fallback_edit_layers),
            n_samples=1,
            n_prompts=merged_n_prompts,
        )
    except StepError as exc:
        record(fallback_label, False, exc)
    except Exception as exc:  # noqa: BLE001
        record(fallback_label, False, f"{type(exc).__name__}: {exc}")
    else:
        record(fallback_label, not problems, "; ".join(problems))

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n[smoke] {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print(f"[smoke] FAILED: {', '.join(failed)}")
        return 1
    print("[smoke] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
