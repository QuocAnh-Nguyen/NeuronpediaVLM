# SPDX-License-Identifier: Apache-2.0
"""Per-layer affine calibration: recovery of a planted affine, byte-identical ``bias=None``
paths, payload round-trips (both schema variants), and the CLI end-to-end.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from jlens.hooks import ActivationRecorder

from vlm_lens._batch import as_batch
from vlm_lens.artifacts import load_bias, load_lens_set, save_bias, save_lens_set
from vlm_lens.calibrate import (
    CALIBRATION_VERSION,
    affine_readout_kwargs,
    bias_payload,
    fit_calibration,
    fit_logit_affines,
    lens_fingerprint,
)
from vlm_lens.data.dummy import build_dummy_manifest
from vlm_lens.evaluate import score_lens
from vlm_lens.fitting import fit_masked
from vlm_lens.positions import build_position_masks
from vlm_lens.models.registry import MODEL_SPECS
from vlm_lens.models.tiny_llava import random_image
from vlm_lens.readout import lens_readout

from .conftest import PROMPT, TINY_CONFIG

LAYER = 0
VOCAB = TINY_CONFIG.vocab_size


def _collect_from(x: torch.Tensor, target: torch.Tensor):
    """A ``fit_logit_affines`` collect closure over materialized logits."""

    def collect(rows: torch.Tensor, layers):
        return {layer: x[rows] for layer in layers}, target[rows]

    return collect


def _split_rows(n: int, *, holdout_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    n_holdout = max(1, int(round(holdout_frac * n)))
    return order[n_holdout:], order[:n_holdout]


def _tiny_samples(tiny_model, seeds) -> list:
    return [
        tiny_model.encode_mm(PROMPT, random_image(seed, image_size=TINY_CONFIG.image_size))
        for seed in seeds
    ]


def test_fit_recovers_planted_scale_and_bias(tiny_model, tiny_lenses, tiny_batch):
    """A known logit-space distortion (scale*x + bias) is recovered from the tiny lens's
    real logits; the hold-out CE strictly decreases.

    ``bias`` is only identified up to an additive constant (softmax is invariant to
    ``+ c * 1``), so the non-constant part is compared after removing the mean - the
    constant is pinned by the mean-residual initialization, not by the loss.
    """
    lens = tiny_lenses["text"]
    base = lens_readout(tiny_model, lens, tiny_batch, layers=[LAYER])
    x = base.lens_logits[LAYER].float()
    assert x.shape[1] == VOCAB

    planted_scale = 1.7
    planted_bias = torch.randn(VOCAB, generator=torch.Generator().manual_seed(7)) * 0.5
    target = planted_scale * x + planted_bias
    train_rows, holdout_rows = _split_rows(int(x.shape[0]), holdout_frac=0.25, seed=0)

    fit = fit_logit_affines(
        _collect_from(x, target),
        layers=[LAYER],
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        epochs=150,
        lr=0.05,
        batch_positions=32,
        patience=0,
        seed=0,
    )
    calibration = fit[LAYER]

    assert calibration.scale == pytest.approx(planted_scale, abs=0.05)
    residual = calibration.bias - planted_bias
    assert float(residual.sub(residual.mean()).abs().max()) < 0.1
    # the fitted affine reproduces the planted distribution on the hold-out rows
    z = calibration.scale * x[holdout_rows] + calibration.bias
    log_q = torch.log_softmax(target[holdout_rows], dim=-1)
    log_p = torch.log_softmax(z, dim=-1)
    kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
    assert float(kl.mean()) < 1e-3
    # hold-out CE strictly decreases against the uncalibrated lens and reaches the
    # target entropy (CE = KL + H(target), and the planted affine is in-family)
    entropy = float(-(log_q.exp() * log_q).sum(dim=-1).mean())
    assert calibration.post["holdout_ce"] < calibration.pre["holdout_ce"]
    assert calibration.post["holdout_ce"] == pytest.approx(entropy + float(kl.mean()), abs=1e-3)

def test_fit_calibration_on_real_tiny_data(tiny_model, tiny_lenses):
    """End-to-end on the tiny fixture: shapes, dtypes, provenance and metric bookkeeping."""
    lens = tiny_lenses["text"]
    samples = _tiny_samples(tiny_model, (12, 13, 14))
    result = fit_calibration(
        tiny_model, lens, samples, epochs=3, lr=0.05, batch_positions=16, seed=0
    )
    assert sorted(result.layers) == list(lens.source_layers)
    for layer, calibration in result.layers.items():
        assert calibration.layer == layer
        assert calibration.bias.dtype == torch.float32
        assert calibration.bias.device.type == "cpu"
        assert calibration.bias.shape == (VOCAB,)
        for metrics in (calibration.pre, calibration.post):
            assert metrics["n"] > 0
            assert metrics["mean_rank_true"] >= 1.0
            assert 0.0 <= metrics["top1_agreement"] <= 1.0
            assert metrics["mean_kl"] >= 0.0
            assert metrics["holdout_ce"] >= 0.0
    assert result.provenance["base_lens"]["fingerprint"] == lens_fingerprint(lens)
    assert result.provenance["n_train_samples"] == 2
    assert result.provenance["n_holdout_samples"] == 1
    assert result.provenance["n_train_positions"] + result.provenance["n_holdout_positions"] > 0
    with pytest.raises(ValueError, match="not fitted"):
        fit_calibration(
            tiny_model,
            lens,
            samples,
            layers=[lens.source_layers[-1] + 1],
            epochs=1,
            lr=0.05,
            batch_positions=8,
        )


def _manual_lens_logits(model, lens, batch, layer: int, positions) -> torch.Tensor:
    """The pre-change lens-logit computation, reimplemented independently."""
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[layer, final_layer]) as recorder:
        model.forward_mm(batch)
    hidden = recorder.activations[layer][0].float()
    seq_len = int(hidden.shape[0])
    index = [p if p >= 0 else seq_len + p for p in positions]
    transported = lens.transport(hidden[index], layer)
    return model.unembed(transported).float().cpu()


def test_none_bias_paths_are_byte_identical(tiny_model, tiny_lenses, tiny_batch):
    """``bias=None`` (and the new ``logit_scale=None``) leave every output bit-identical
    to the pre-change path, which the test re-derives independently."""
    lens = tiny_lenses["text"]
    positions = [0, 2, 5, -1]
    baseline = lens_readout(tiny_model, lens, tiny_batch, layers=[LAYER], positions=positions)
    with_none = lens_readout(
        tiny_model,
        lens,
        tiny_batch,
        layers=[LAYER],
        positions=positions,
        bias=None,
        scale=None,
        temp=None,
        logit_scale=None,
        logit_bias=None,
    )
    assert torch.equal(with_none.lens_logits[LAYER], baseline.lens_logits[LAYER])
    assert torch.equal(with_none.model_logits, baseline.model_logits)
    manual = _manual_lens_logits(tiny_model, lens, tiny_batch, LAYER, positions)
    assert torch.equal(baseline.lens_logits[LAYER], manual)

    plain = score_lens(tiny_model, lens, [tiny_batch], tags=("text",))
    explicit = score_lens(tiny_model, lens, [tiny_batch], tags=("text",), bias=None, scale=None)
    assert plain == explicit


def test_score_lens_applies_calibration(tiny_model, tiny_lenses, tiny_batch):
    """``score_lens(bias=..., scale=...)`` scores the calibrated logits: the metrics equal
    an independent brute force over the same calibrated rows and differ from the plain lens."""
    lens = tiny_lenses["text"]
    scale_map = {LAYER: 1.7}
    bias_map = {LAYER: torch.randn(VOCAB, generator=torch.Generator().manual_seed(3))}

    scores = {
        row.layer: row
        for row in score_lens(
            tiny_model, lens, [tiny_batch], tags=("text",), bias=bias_map, scale=scale_map
        )
    }
    plain = {row.layer: row for row in score_lens(tiny_model, lens, [tiny_batch], tags=("text",))}
    assert LAYER in scores

    # brute force over the text positions: calibrated lens logits vs the model's
    batch = as_batch(tiny_model, tiny_batch, 1536)
    masks = build_position_masks(batch.input_ids, tiny_model.image_token_id, masks=("text",))
    positions = [
        int(p)
        for p in masks["text"].nonzero(as_tuple=True)[0]
        if int(p) + 1 < batch.seq_len and int(batch.input_ids[0, int(p) + 1]) != tiny_model.image_token_id
    ]
    readout = lens_readout(
        tiny_model, lens, batch, layers=[LAYER], positions=positions,
        logit_bias=bias_map, logit_scale=scale_map,
    )
    targets = batch.input_ids[0][[p + 1 for p in positions]]
    rows = readout.lens_logits[LAYER]
    model_rows = readout.model_logits
    top1 = float((rows.argmax(dim=1) == model_rows.argmax(dim=1)).float().mean())
    kl = float(
        torch.nn.functional.kl_div(
            torch.log_softmax(rows, dim=-1),
            torch.log_softmax(model_rows, dim=-1),
            reduction="none",
            log_target=True,
        ).sum(dim=-1).mean()
    )
    assert scores[LAYER].n == len(positions)
    assert scores[LAYER].top1_agreement == pytest.approx(top1, rel=1e-4)
    assert scores[LAYER].mean_kl == pytest.approx(kl, rel=1e-4, abs=1e-6)
    # the corrections actually moved the metrics
    assert scores[LAYER].mean_kl != pytest.approx(plain[LAYER].mean_kl)


def test_bias_payload_round_trip_and_schema_variants(tiny_model, tiny_lenses, tiny_batch, tmp_path):
    """``save_bias``/``load_bias`` round-trip the calibration payload (fingerprint in
    meta), both payload schema variants load, and the fitted affine applied through
    ``lens_readout``'s ``logit_scale``/``logit_bias`` reproduces the calibration."""
    lens = tiny_lenses["text"]
    samples = _tiny_samples(tiny_model, (12, 13, 14))
    result = fit_calibration(
        tiny_model, lens, samples, epochs=2, lr=0.05, batch_positions=16, seed=0
    )
    payload = bias_payload(result)
    assert payload["meta"]["calibration_version"] == CALIBRATION_VERSION

    written = save_bias(tmp_path / "bias-text.pt", payload)
    loaded, meta = load_bias(written)
    assert meta["base_lens"]["fingerprint"] == lens_fingerprint(lens)
    d_model = int(result.provenance["d_model"])
    for layer, calibration in result.layers.items():
        entry = loaded[str(layer)]
        assert torch.equal(entry["logit_bias"], calibration.bias)
        assert float(entry["logit_scale"]) == calibration.scale
        assert torch.equal(entry["bias"], torch.zeros(d_model))
        assert float(entry["scale"]) == 1.0

    # the consumer helper maps the payload to lens_readout kwargs for both variants
    kwargs = affine_readout_kwargs(loaded)
    for layer, calibration in result.layers.items():
        assert torch.equal(kwargs["logit_bias"][layer], calibration.bias)
        assert kwargs["logit_scale"][layer] == calibration.scale
    old_payload = {
        str(LAYER): {"bias": torch.zeros(d_model), "scale": 1.5},
        "meta": {"kind": "moment-census"},
    }
    assert affine_readout_kwargs(old_payload) == {"logit_bias": {}, "logit_scale": {}}
    old_written = save_bias(tmp_path / "bias-plain.pt", old_payload)
    loaded_old, _ = load_bias(old_written)
    assert float(loaded_old[str(LAYER)]["scale"]) == 1.5
    assert affine_readout_kwargs(loaded_old) == {"logit_bias": {}, "logit_scale": {}}

    # applying the loaded payload through the readout reproduces the calibrated logits
    baseline = lens_readout(tiny_model, lens, tiny_batch, layers=[LAYER], positions=[-1])
    calibrated = lens_readout(
        tiny_model, lens, tiny_batch, layers=[LAYER], positions=[-1], **kwargs
    )
    calibration = result.layers[LAYER]
    assert torch.equal(
        calibrated.lens_logits[LAYER],
        baseline.lens_logits[LAYER] * calibration.scale + calibration.bias,
    )


def test_fit_calibration_cli_tiny_backend(tmp_path):
    """The CLI fits and writes ``bias-<mask>.pt`` end-to-end on the tiny backend."""
    spec = importlib.util.spec_from_file_location(
        "fit_calibration", Path(__file__).resolve().parents[1] / "scripts" / "fit_calibration.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_spec = MODEL_SPECS["llava-7b"]
    hf_model, processor = model_spec.build_tiny(image_size=56)
    model = model_spec.lens_model_cls()(hf_model, processor)
    samples = [
        model.encode_mm(PROMPT, random_image(seed, image_size=56)) for seed in (20, 21, 22)
    ]
    fitted = fit_masked(
        model, samples, source_layers=[0, 1], dim_batch=8, skip_first=1, masks=("text",)
    )
    lens_dir = tmp_path / "artifacts"
    save_lens_set(lens_dir, fitted.lenses, provenance={"note": "cli test"})
    loaded_lenses, _ = load_lens_set(lens_dir)
    manifest = build_dummy_manifest(tmp_path / "corpus", n_samples=3, image_size=56, seed=0)

    out_dir = tmp_path / "calibration"
    rc = module.main(
        [
            "--model",
            "llava-7b",
            "--backend",
            "tiny",
            "--tiny-image-size",
            "56",
            "--lens-dir",
            str(lens_dir),
            "--mask",
            "text",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-positions",
            "16",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    bias_path = out_dir / "bias-text.pt"
    assert bias_path.is_file()
    loaded, meta = load_bias(bias_path)
    # the fingerprint binds to the lens *as read from disk* (fp16-rounded J values),
    # which is the content the calibration actually ran against
    assert meta["base_lens"]["fingerprint"] == lens_fingerprint(loaded_lenses["text"])
    assert meta["backend"] == "tiny"
    assert meta["manifest_sha256"]
    for layer in (0, 1):
        assert isinstance(loaded[str(layer)]["logit_bias"], torch.Tensor)
        assert "logit_scale" in loaded[str(layer)]
