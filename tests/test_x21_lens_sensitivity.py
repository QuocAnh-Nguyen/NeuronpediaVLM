# SPDX-License-Identifier: Apache-2.0
"""x21's report metrics are strictly the loaded lens's: different J -> different report.

Regression contract behind the ``x21_report_qwen2vl.json`` vs ``_64tok.json``
investigation (two runs reported bit-identical metrics): the report's Spearman
readout and its coordinate swaps must consume the fitted ``J_l`` from ``--lens-dir``
-- no aliasing to the model's next-token logits, no silent identity/logit-lens
fallback -- so a genuinely different lens cannot produce a bit-identical report, and
``meta.lens.fingerprint`` must certify which lens *content* was read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "results" / "validation_2026-10-01" / "code"
for _path in (str(REPO / "src"), str(REPO / "third_party" / "jacobian-lens"), str(CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import x21_verbal_report as x21  # noqa: E402  (after the sys.path bootstrap, as in the script)
from jlens.lens import JacobianLens  # noqa: E402
import vlm_registry as registry  # noqa: E402


def tiny_args(lens_dir: str | Path, **over) -> argparse.Namespace:
    """The report arm's args on the tiny fixture (fast knobs, real defaults otherwise)."""
    base = dict(
        arm="report",
        backend="tiny",
        model="llava-7b",
        lens_dir=str(lens_dir),
        mask="text",
        bias_dir="",
        calibrated=False,
        experiments_dir=str(x21.DEFAULT_EXPERIMENTS_DIR),
        layers=None,
        layer_lo=None,
        layer_hi=None,
        max_new_tokens=2,
        max_seq_len=1536,
        json="/dev/null",
        n_candidates=6,
        n_swaps=2,
        swap_alpha=1.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


def synth_lens(model, perturbation: float, seed: int) -> JacobianLens:
    """The tiny geometry's lens: identity (``perturbation=0``) or identity + a seeded kick."""
    device = model.unembed_weight().device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    jacobians = {}
    for layer in range(model.n_layers):
        J = torch.eye(model.d_model, dtype=torch.float32)
        if perturbation:
            R = torch.randn(model.d_model, model.d_model, generator=gen, dtype=torch.float32)
            J = J + perturbation * R / R.norm() * (model.d_model ** 0.5)
        jacobians[layer] = J.to(device)
    return JacobianLens(jacobians=jacobians, n_prompts=0, d_model=model.d_model)


def write_lens_dir(path: Path, lens: JacobianLens) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    lens.save(str(path / "lens-text.pt"))
    return path


@pytest.fixture(scope="module")
def tiny_model():
    return registry.load_lens_model(tiny_args("/nonexistent"))


@pytest.fixture(scope="module")
def lens_pair(tmp_path_factory, tiny_model):
    """Two genuinely different ``--lens-dir`` trees: identity vs identity + 10% kick."""
    root = tmp_path_factory.mktemp("lens-dirs")
    lens_a = synth_lens(tiny_model, 0.0, 1)
    lens_b = synth_lens(tiny_model, 0.1, 1234)
    return (
        write_lens_dir(root / "lensA", lens_a),
        write_lens_dir(root / "lensB", lens_b),
        lens_a,
        lens_b,
    )


def test_load_lens_reads_content_and_fingerprints_it(tiny_model, lens_pair):
    dir_a, dir_b, lens_a, lens_b = lens_pair
    lens_loaded, meta_a = x21.load_lens(tiny_args(dir_a), tiny_model)
    assert meta_a["loaded"] is True
    assert meta_a["dir"] == str(dir_a)
    assert meta_a["source_layers"] == list(lens_a.source_layers)
    _lens_b, meta_b = x21.load_lens(tiny_args(dir_b), tiny_model)
    assert meta_b["loaded"] is True
    assert meta_a["fingerprint"] == x21.lens_fingerprint(lens_a)
    assert meta_a["fingerprint"] != meta_b["fingerprint"], "different J must fingerprint apart"


def test_load_lens_tiny_falls_back_with_reason(tiny_model, tmp_path):
    missing = tmp_path / "does-not-exist"
    lens, meta = x21.load_lens(tiny_args(missing), tiny_model)
    assert meta["loaded"] is False
    assert "unreadable" in meta["fallback"]
    assert meta["fingerprint"] == x21.lens_fingerprint(lens)
    assert meta["source_layers"] == list(range(tiny_model.n_layers))
    wrong_mask = tmp_path / "wrong-mask"
    write_lens_dir(wrong_mask, synth_lens(tiny_model, 0.0, 1))
    _lens, meta = x21.load_lens(tiny_args(wrong_mask, mask="image"), tiny_model)
    assert meta["loaded"] is False
    assert "mask 'image' not in" in meta["fallback"]


def test_readouts_refuse_unfitted_layers(tiny_model, lens_pair):
    _dir_a, _dir_b, lens_a, _lens_b = lens_pair
    args = tiny_args("/nonexistent")
    kind = registry.kind_for(args)
    model = tiny_model
    layers, _note = x21.resolve_layers(args, lens_a, kind)
    batch = x21.encode_chat(model, kind, x21.REPORT_TEMPLATE.format(category="animal"), 1536)
    _tokens, records, _scores = x21.trace_generation(model, batch, layers, max_new_tokens=1)
    gap = synth_lens(model, 0.0, 1)
    gap.jacobians.pop(layers[-1])  # one band layer unfitted
    with pytest.raises(ValueError, match="not among the loaded lens's fitted layers"):
        x21.lens_logits(model, gap, records[0][layers[-1]], layers[-1], None)
    with pytest.raises(ValueError, match="not among the loaded lens's fitted layers"):
        x21.vectors_for_ids(model, gap, [1, 2], [layers[-1]])


def test_report_metrics_follow_the_loaded_lens(tiny_model, lens_pair):
    dir_a, dir_b, _lens_a, _lens_b = lens_pair
    args = tiny_args(dir_a)
    model = tiny_model
    lens_a, meta_a = x21.load_lens(args, model)
    lens_b, meta_b = x21.load_lens(tiny_args(dir_b), model)
    layers, _note = x21.resolve_layers(args, lens_a, registry.kind_for(args))

    out_a = x21.arm_report(args, model, lens_a, None, layers)
    out_b = x21.arm_report(args, model, lens_b, None, layers)
    # main() injects the loaded-lens meta into the report before writing it.
    out_a["meta"]["lens"] = meta_a
    out_b["meta"]["lens"] = meta_b

    # (a) Spearman readout: the per-layer lens-vs-model correlations move with the lens.
    assert out_a["per_layer"] != out_b["per_layer"]
    moved = [
        layer
        for layer in layers
        if out_a["per_layer"][str(layer)]["mean_spearman_per_category"]
        != out_b["per_layer"][str(layer)]["mean_spearman_per_category"]
    ]
    assert moved, "a different fitted lens left every reported Spearman rho unchanged"

    # (b) Swaps: same targets (selection is lens-independent), different patched outcomes.
    swaps_a = [s for category in out_a["categories"] for s in category["swaps"]]
    swaps_b = [s for category in out_b["categories"] for s in category["swaps"]]
    assert [s["target"] for s in swaps_a] == [s["target"] for s in swaps_b]
    assert swaps_a != swaps_b, "the swap path ignored the fitted lens"

    # The fingerprint certifies which lens content each report read.
    assert out_a["meta"]["lens"]["fingerprint"] == meta_a["fingerprint"]
    assert out_b["meta"]["lens"]["fingerprint"] == meta_b["fingerprint"]
    assert meta_a["fingerprint"] != meta_b["fingerprint"]
