# SPDX-License-Identifier: Apache-2.0
"""Readout, generation tracing, and J-lens-vector interventions."""

from __future__ import annotations

import torch
from jlens.hooks import ActivationRecorder

from vlm_lens.interventions import (
    ResidualEdit,
    ResidualEditor,
    apply_edit,
    generate_with_edits,
    lens_vectors,
)
from vlm_lens.readout import lens_readout, trace_generation

LAYER = 1


def test_lens_readout_shapes_and_metadata(tiny_model, tiny_lenses, tiny_batch):
    lens = tiny_lenses["text"]
    readout = lens_readout(tiny_model, lens, tiny_batch, layers=[0, LAYER], positions=[-1, -2, 5])
    assert set(readout.lens_logits) == {0, LAYER}
    for layer in (0, LAYER):
        assert readout.lens_logits[layer].shape == (
            3,
            tiny_model.hf_model.config.text_config.vocab_size,
        )
    assert readout.model_logits.shape[0] == 3
    assert readout.tags == ["text", "text", "image"]
    assert all(tag in {"image", "text"} for tag in readout.tags)

    table = readout.top_k(k=3)
    assert set(table) == {0, LAYER}
    assert len(table[LAYER]) == 3 and len(table[LAYER][0]) == 3
    token, probability = table[LAYER][0][0]
    assert isinstance(token, str) and 0.0 <= probability <= 1.0

    # logit-lens baseline (no J transport) also returns well-formed logits
    baseline = lens_readout(tiny_model, lens, tiny_batch, layers=[LAYER], positions=[-1], use_jacobian=False)
    assert baseline.lens_logits[LAYER].shape[0] == 1
    assert not torch.allclose(baseline.lens_logits[LAYER], readout.lens_logits[LAYER][2:3])


def test_image_positions_are_tagged(tiny_model, tiny_lenses, tiny_batch):
    readout = lens_readout(
        tiny_model, tiny_lenses["all"], tiny_batch, layers=[0], positions=[2, 3, -1]
    )
    assert readout.tags == ["image", "image", "text"]


def test_readout_final_layer_matches_model_logits(tiny_model, tiny_lenses, tiny_batch):
    """Reading out the last block is the identity transport: must equal HF logits."""
    readout = lens_readout(tiny_model, tiny_lenses["text"], tiny_batch, layers=[], positions=[-1])
    hf_logits = tiny_model.hf_model(**tiny_batch.hf_kwargs(), use_cache=False).logits[0, -1]
    torch.testing.assert_close(readout.model_logits[0], hf_logits.float().cpu(), rtol=1e-4, atol=1e-5)


def test_trace_generation_aligns_steps_with_tokens(tiny_model, tiny_lenses, tiny_batch):
    trace = trace_generation(
        tiny_model, tiny_lenses["text"], tiny_batch, layers=[0, LAYER], max_new_tokens=3, top_k=2
    )
    assert len(trace["steps"]) == len(trace["token_ids"]) == 3
    for step in trace["steps"]:
        assert set(step) == {0, LAYER, tiny_model.n_layers - 1}
        assert len(step[LAYER]["top_tokens"]) == 2
    assert isinstance(trace["text"], str)


def test_apply_edit_math():
    d_model = 6
    torch.manual_seed(0)
    v_s = torch.randn(d_model)
    v_t = torch.randn(d_model)
    # h lives in the span of [v_s, v_t] with known coordinates: 3 * v_s + 5 * v_t
    h = (3.0 * v_s + 5.0 * v_t).reshape(1, 1, d_model)
    vectors = {"source": v_s, "target": v_t}

    swapped = apply_edit(h, ResidualEdit(layer=0, mode="swap", token="t", source_token="s"), vectors)
    expected = (5.0 * v_s + 3.0 * v_t).reshape(1, 1, d_model)
    torch.testing.assert_close(swapped, expected, rtol=1e-4, atol=1e-4)

    added = apply_edit(h, ResidualEdit(layer=0, mode="add", alpha=2.0, token="t"), vectors)
    torch.testing.assert_close(added, h + 2.0 * v_t, rtol=0, atol=0)

    ablated = apply_edit(h, ResidualEdit(layer=0, mode="ablate", token="t"), vectors)
    assert abs(float((ablated * (v_t / v_t.norm())).sum())) < 1e-5
    # what was removed is exactly the projection along v_t (orthogonal complement intact)
    unit = v_t / v_t.norm()
    removed = h - ablated
    torch.testing.assert_close(removed, (h * unit).sum() * unit.reshape(1, 1, -1), rtol=1e-4, atol=1e-5)


def test_lens_vectors_are_readout_gradients(tiny_model, tiny_lenses):
    """``v_t`` must be the gradient of the lens readout ``W_U[t] . (J_l h)`` w.r.t. ``h``.

    Guards the orientation of ``W_U @ J``: a transposed ``J`` yields a vector of similar
    scale pointing elsewhere, which no mechanics-only test would notice.
    """
    lens = tiny_lenses["text"]
    ids = tiny_model.tokenizer("dog", add_special_tokens=False)["input_ids"]
    token_id = int(torch.as_tensor(ids).reshape(-1)[-1])  # mirror _token_id: last id
    vectors = lens_vectors(tiny_model, lens, ["dog"], layers=[LAYER])[LAYER][0]

    weight = tiny_model.unembed_weight().detach().float()[token_id]
    h = torch.zeros(tiny_model.d_model, requires_grad=True)  # linear map: gradient is h-free
    (lens.transport(h.unsqueeze(0), LAYER) * weight).sum().backward()
    torch.testing.assert_close(vectors, h.grad, rtol=1e-4, atol=1e-6)

def test_residual_editor_applies_exact_delta(tiny_model, tiny_lenses, tiny_batch):
    vectors = lens_vectors(tiny_model, tiny_lenses["text"], ["dog", "cat"], layers=[LAYER])
    direction = vectors[LAYER][0]
    edit = ResidualEdit(layer=LAYER, mode="add", alpha=0.75, token="dog", positions="all")

    with ActivationRecorder(tiny_model.layers, at=[LAYER]) as recorder:
        tiny_model.forward_mm(tiny_batch)
        clean = recorder.activations[LAYER].detach().clone()

    with ResidualEditor(tiny_model, tiny_lenses["text"], [edit]):
        with ActivationRecorder(tiny_model.layers, at=[LAYER]) as recorder:
            tiny_model.forward_mm(tiny_batch)
            edited = recorder.activations[LAYER].detach().clone()

    delta = edited - clean
    assert delta.shape == clean.shape
    # every position got the same added direction: delta / alpha == direction (broadcast)
    torch.testing.assert_close(delta, 0.75 * direction.reshape(1, 1, -1).expand_as(delta), rtol=1e-3, atol=1e-4)


def test_residual_editor_positions_last_only(tiny_model, tiny_lenses, tiny_batch):
    edit = ResidualEdit(layer=LAYER, mode="ablate", token="dog", positions="last")
    with ActivationRecorder(tiny_model.layers, at=[LAYER]) as recorder:
        tiny_model.forward_mm(tiny_batch)
        clean = recorder.activations[LAYER].detach().clone()
    with ResidualEditor(tiny_model, tiny_lenses["text"], [edit]):
        with ActivationRecorder(tiny_model.layers, at=[LAYER]) as recorder:
            tiny_model.forward_mm(tiny_batch)
            edited = recorder.activations[LAYER].detach().clone()
    assert torch.allclose(clean[:, :-1], edited[:, :-1])
    assert not torch.allclose(clean[:, -1], edited[:, -1])


def test_generate_with_edits_changes_logits_and_runs(tiny_model, tiny_lenses, tiny_batch):
    baseline = generate_with_edits(tiny_model, None, tiny_batch, max_new_tokens=2)
    assert baseline["edits"] == [] and baseline["n_edit_forwards"] == 0

    big = lens_vectors(tiny_model, tiny_lenses["text"], ["dog"], layers=[LAYER])[LAYER][0]
    alpha = 50.0 / float(big.norm())  # deliberately large so the effect is unambiguous
    edits = [ResidualEdit(layer=LAYER, mode="add", alpha=alpha, token="dog", positions="all")]
    edited = generate_with_edits(tiny_model, tiny_lenses["text"], tiny_batch, edits=edits, max_new_tokens=2)
    assert edited["n_edit_forwards"] > 0
    # the edit flips the first step to EOS, which legitimately truncates generation
    assert 1 <= len(edited["token_ids"]) <= 2

    # the same edits must change a single forward's logits
    def logits(use_edits: bool) -> torch.Tensor:
        context = (
            ResidualEditor(tiny_model, tiny_lenses["text"], edits) if use_edits else ResidualEditor(tiny_model, tiny_lenses["text"], [])
        )
        with context:
            return tiny_model.hf_model(**tiny_batch.hf_kwargs(), use_cache=False).logits

    assert not torch.allclose(logits(False), logits(True))


def test_lens_readout_final_layer_is_the_model_logits(tiny_model, tiny_lenses, tiny_batch):
    """The docstring promises the final layer is always available; it is the logit lens."""
    final = tiny_model.n_layers - 1
    assert final not in tiny_lenses["text"].source_layers
    readout = lens_readout(tiny_model, tiny_lenses["text"], tiny_batch, layers=[final], positions=[-1])
    assert set(readout.lens_logits) == {final}
    assert torch.allclose(readout.lens_logits[final], readout.model_logits)


def test_apply_edit_aligns_vectors_to_the_residual(tiny_model, tiny_lenses):
    """`lens_vectors` returns CPU fp32; residuals are bf16/CUDA on the cluster (real bug)."""
    vector = lens_vectors(tiny_model, tiny_lenses["text"], ["dog"], layers=[LAYER])[LAYER][0]
    hidden = torch.randn(2, vector.numel(), dtype=torch.float64)
    edit = ResidualEdit(layer=LAYER, mode="add", alpha=0.5, token="dog")
    out = apply_edit(hidden, edit, {"target": vector})
    assert out.dtype == torch.float64
    assert torch.allclose(out, hidden + 0.5 * vector.to(torch.float64), atol=1e-6)
