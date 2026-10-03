# SPDX-License-Identifier: Apache-2.0
"""HTTP tests for the demo backend on the CPU tiny model (no network, no GPU).

Run from ``apps/vlm-jlens-demo/backend``::

    ~/miniforge3/envs/vlm-lens/bin/python -m pytest tests/test_api_tiny.py -v

Every app is built from an explicit :class:`~vlmj.engine.Config`, so the tests
never read ``VLMJ_*`` from the ambient environment and never download weights.
Caches and mock lenses are written under ``tmp_path_factory`` only.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

MAX_NEW_TOKENS_CAP = 64
GEN_TOKENS = 4
BACKEND_ROOT = Path(__file__).resolve().parents[1]

#: ``{layer: {prob, rank, logit}}``-style keys of one top-k entry.
TOPK_KEYS = {"str", "prob", "logit", "rank"}


def _poll_job(client: Any, job_id: str, *, timeout: float = 300.0) -> dict[str, Any]:
    """Block until the job leaves the queue, then return its snapshot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        snapshot = response.json()
        if snapshot["status"] in ("done", "error"):
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _run_job(client: Any, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST a job request and return its finished snapshot."""
    response = client.post(path, json=body)
    assert response.status_code == 200, f"{path}: {response.status_code} {response.text}"
    return _poll_job(client, response.json()["job_id"])


@contextlib.contextmanager
def _build_client(tmp_path: Path, **overrides: Any) -> Iterator[Any]:
    """Tiny app (explicit config, tmp cache) as a live ``TestClient`` context."""
    from fastapi.testclient import TestClient

    from vlmj.app import create_app
    from vlmj.engine import Config

    config = Config(backend="tiny", cache_dir=tmp_path, **overrides)
    with TestClient(create_app(config)) as client:
        yield client


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """Tiny backend with the auto-fitted lens (fits once for the whole module)."""
    with _build_client(tmp_path_factory.mktemp("vlmj-autofit")) as tiny_client:
        yield tiny_client


@pytest.fixture(scope="module")
def session(client: Any) -> dict[str, Any]:
    """One encoded sample image + the default prompt."""
    samples = client.get("/api/samples").json()["samples"]
    response = client.post("/api/session", json={"image_b64": samples[0]["image_b64"]})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def generated(client: Any, session: dict[str, Any]) -> dict[str, Any]:
    """Result of the baseline generation, shared by the intervention tests."""
    job = _run_job(
        client,
        "/api/generate",
        {"session_id": session["session_id"], "max_new_tokens": GEN_TOKENS},
    )
    assert job["status"] == "done", job
    return job["result"]


# --------------------------------------------------------------------------- meta


def test_meta(client: Any) -> None:
    payload = client.get("/api/meta").json()
    assert set(payload) == {"mode", "device", "dtype", "model", "lens", "capabilities", "jobs", "gpu"}
    assert payload["mode"] == "tiny"
    assert payload["device"] == "cpu"
    assert payload["dtype"] == "float32"
    model = payload["model"]
    assert model == {
        "name": "tiny-llava-demo",
        "n_layers": 3,
        "d_model": 16,
        "image_seq_length": 576,
        "grid": [24, 24],
        "vocab_size": 64,
    }
    assert payload["capabilities"] == {
        "lens_readout": True,
        "attribution_metrics": ["lens_prob", "lens_logit", "attn_rollout"],
        "knockout": True,
        "steer_modes": ["add", "ablate", "swap"],
        "max_new_tokens": MAX_NEW_TOKENS_CAP,
    }
    lens = payload["lens"]
    assert lens["available"] is True
    assert lens["mock"] is False
    assert lens["source_layers"] == [0, 1]
    assert sorted(lens["masks"]) == ["all", "image", "text"]
    assert set(lens["n_prompts"]) == {"all", "image", "text"}
    assert set(lens["n_prompts"].values()) == {3}
    assert "not meaningful" in (lens["notes"] or "")
    assert lens["error"] is None
    assert payload["jobs"] == {"active": 0, "queued": 0}
    assert payload["gpu"] is None or set(payload["gpu"]) == {"free_mb", "total_mb"}


def test_samples(client: Any) -> None:
    samples = client.get("/api/samples").json()["samples"]
    assert len(samples) == 3
    assert len({sample["id"] for sample in samples}) == 3
    for sample in samples:
        assert sample["label"]
        assert sample["image_b64"].startswith("data:image/png;base64,")


# --------------------------------------------------------------------------- sessions


def test_session(client: Any, session: dict[str, Any]) -> None:
    assert set(session) == {
        "session_id",
        "mode",
        "model",
        "lens",
        "variant",
        "prompt_text",
        "text_tokens",
        "image",
        "has_baseline",
    }
    assert session["has_baseline"] is False
    assert session["variant"] == "image"
    image = session["image"]
    assert image["grid"] == [24, 24]
    assert image["count"] == 576
    assert (image["start"], image["end"]) == (2, 578)  # end is exclusive
    assert image["end"] - image["start"] == image["count"]
    quarters = image["quarter_of_patch"]
    assert len(quarters) == 576
    assert sorted(set(quarters)) == [0, 1, 2, 3]
    assert len(session["text_tokens"]) == 5
    assert [token["i"] for token in session["text_tokens"]] == [0, 1, 578, 579, 580]
    assert "Describe this image" in session["prompt_text"]

    info = client.get(f"/api/session/{session['session_id']}").json()
    assert info["has_baseline"] is False and info["baseline_caption"] is None
    assert info["session_id"] == session["session_id"]
    assert info["lens"]["available"] is True
    assert info["created_utc"]


def test_unknown_session_and_job(client: Any) -> None:
    assert client.get("/api/session/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


# --------------------------------------------------------------------------- generate


def test_generate(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    assert set(generated) == {"caption", "tokens", "prompt_len"}
    assert generated["prompt_len"] == 581
    assert len(generated["tokens"]) <= GEN_TOKENS
    assert generated["tokens"], "the tiny model should emit at least one token"
    for index, token in enumerate(generated["tokens"]):
        assert token["i"] == index
        assert isinstance(token["id"], int) and 0 <= token["id"] < 64
        assert isinstance(token["str"], str) and token["str"]
        assert token["logprob"] <= 0.0
    assert isinstance(generated["caption"], str)

    info = client.get(f"/api/session/{session['session_id']}").json()
    assert info["has_baseline"] is True
    assert info["baseline_caption"] == generated["caption"]


# --------------------------------------------------------------------------- lens


def test_lens(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    tracked = generated["tokens"][0]["str"]
    job = _run_job(
        client,
        "/api/lens",
        {
            "session_id": session["session_id"],
            "layers": [0, 1],
            "targets": [
                {"kind": "gen", "i": 0},
                {"kind": "prompt", "i": 0},
                {"kind": "patch", "i": 300},
            ],
            "topk": 3,
            "track": [tracked],
        },
    )
    assert job["status"] == "done", job
    result = job["result"]
    assert result["vocab_size"] == 64
    assert [target["kind"] for target in result["targets"]] == ["gen", "prompt", "patch"]
    assert all(target["label"] for target in result["targets"])
    for target in result["targets"]:
        assert [entry["layer"] for entry in target["per_layer"]] == [0, 1]
        for entry in target["per_layer"]:
            assert len(entry["topk"]) == 3
            probs = [item["prob"] for item in entry["topk"]]
            ranks = [item["rank"] for item in entry["topk"]]
            assert ranks == sorted(ranks) and ranks[0] == 1  # rank 1 is the top token
            assert all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1))
            assert all(set(item) == TOPK_KEYS for item in entry["topk"])
            tracked_entry = entry["tracked"][tracked]
            assert set(tracked_entry) == {"prob", "rank", "logit"}
            assert tracked_entry["rank"] >= 1
            assert 0.0 <= tracked_entry["prob"] <= 1.0


def test_lens_model_row(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    """``model_row`` is the model's own next-token distribution at each target."""
    tracked = generated["tokens"][0]["str"]
    job = _run_job(
        client,
        "/api/lens",
        {
            "session_id": session["session_id"],
            "layers": [0],
            "targets": [{"kind": "gen", "i": 0}, {"kind": "prompt", "i": 1}, {"kind": "patch", "i": 42}],
            "topk": 4,
            "track": [tracked],
        },
    )
    assert job["status"] == "done", job
    targets = job["result"]["targets"]
    assert [target["kind"] for target in targets] == ["gen", "prompt", "patch"]
    for target in targets:
        row = target["model_row"]
        assert set(row) == {"topk", "tracked"}
        assert len(row["topk"]) == 4
        assert all(set(item) == TOPK_KEYS for item in row["topk"])
        probs = [item["prob"] for item in row["topk"]]
        ranks = [item["rank"] for item in row["topk"]]
        assert ranks == sorted(ranks) and ranks[0] == 1  # rank 1 is the top token
        assert all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1))
        assert all(0.0 <= prob <= 1.0 for prob in probs)
        tracked_entry = row["tracked"][tracked]
        assert set(tracked_entry) == {"prob", "rank", "logit"}
        assert tracked_entry["rank"] >= 1

    # The model's own row, not the lens transport: greedy decoding picked the argmax of
    # exactly this distribution, so its top-1 and logprob must match the caption tokens.
    first_generated = generated["tokens"][0]
    gen_row = targets[0]["model_row"]
    assert gen_row["topk"][0]["str"] == first_generated["str"]
    assert math.log(gen_row["tracked"][first_generated["str"]]["prob"]) == pytest.approx(
        first_generated["logprob"], abs=1e-3
    )


# --------------------------------------------------------------------------- attribution


def _check_attribution(payload: dict[str, Any], session: dict[str, Any]) -> None:
    assert payload["grid"] and all(len(row) == 24 for row in payload["grid"])
    assert len(payload["grid"]) == 24
    values = [value for row in payload["grid"] for value in row]
    assert all(value == value for value in values)  # no NaN
    assert payload["vmin"] == min(values) and payload["vmax"] == max(values)
    assert payload["vmax"] >= payload["vmin"]
    quarters = payload["quarters"]
    assert len(quarters) == 4
    for quarter in range(4):
        members = [
            payload["grid"][patch // 24][patch % 24]
            for patch, index in enumerate(session["image"]["quarter_of_patch"])
            if index == quarter
        ]
        assert members
        assert quarters[quarter] == pytest.approx(sum(members) / len(members), rel=1e-6, abs=1e-9)


def test_attribution_lens_prob(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    job = _run_job(
        client,
        "/api/attribution",
        {
            "session_id": session["session_id"],
            "layer": 0,
            "token": generated["tokens"][0]["str"],
            "metric": "lens_prob",
        },
    )
    assert job["status"] == "done", job
    payload = job["result"]
    assert payload["layer"] == 0
    assert payload["metric"] == "lens_prob"
    assert payload["rollout"] is False
    _check_attribution(payload, session)
    assert 0.0 <= payload["vmin"] <= payload["vmax"] <= 1.0


def test_attribution_attn_rollout(client: Any, session: dict[str, Any]) -> None:
    job = _run_job(
        client,
        "/api/attribution",
        {
            "session_id": session["session_id"],
            "layer": 1,
            "token": "",  # unused for attention rollout
            "metric": "attn_rollout",
            "rollout": True,
        },
    )
    assert job["status"] == "done", job
    payload = job["result"]
    assert payload["metric"] == "attn_rollout" and payload["rollout"] is True
    _check_attribution(payload, session)
    assert payload["grid_rank"] is None


def test_attribution_grid_rank(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    token = generated["tokens"][0]["str"]
    job = _run_job(
        client,
        "/api/attribution",
        {
            "session_id": session["session_id"],
            "layer": 0,
            "token": token,
            "metric": "lens_prob",
        },
    )
    assert job["status"] == "done", job
    payload = job["result"]
    ranks = payload["grid_rank"]
    assert len(ranks) == 24 and all(len(row) == 24 for row in ranks)
    assert all(isinstance(rank, int) and rank >= 1 for row in ranks for rank in row)
    flat_grid = [value for row in payload["grid"] for value in row]
    best = max(range(len(flat_grid)), key=flat_grid.__getitem__)

    # `grid_rank` and the lens readout's `tracked` rank the same lens logits, so the
    # patch with the largest probability must agree with a direct readout there.
    readout = _run_job(
        client,
        "/api/lens",
        {
            "session_id": session["session_id"],
            "layers": [0],
            "targets": [{"kind": "patch", "i": best}],
            "topk": 64,
            "track": [token],
        },
    )
    assert readout["status"] == "done", readout
    entry = readout["result"]["targets"][0]["per_layer"][0]
    tracked_rank = entry["tracked"][token]["rank"]
    assert ranks[best // 24][best % 24] == tracked_rank
    index_of_token = next(i for i, item in enumerate(entry["topk"]) if item["str"] == token)
    assert entry["topk"][index_of_token]["rank"] == tracked_rank
    # rank 1 at a patch is exactly "the requested token is that patch's argmax".
    assert (tracked_rank == 1) == (index_of_token == 0)


# --------------------------------------------------------------------------- knockout


def test_knockout(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    patches = list(range(16))
    job = _run_job(
        client,
        "/api/knockout",
        {
            "session_id": session["session_id"],
            "patches": patches,
            "mode": "zero",
            "max_new_tokens": GEN_TOKENS,
        },
    )
    assert job["status"] == "done", job
    payload = job["result"]
    assert payload["patches"] == patches
    assert payload["mode"] == "zero"
    assert payload["baseline_caption"] == generated["caption"]
    assert isinstance(payload["caption_after"], str)
    assert len(payload["deltas"]) == len(generated["tokens"])
    for token, entry in zip(generated["tokens"], payload["deltas"]):
        assert set(entry) == {"token", "logprob_before", "logprob_after", "delta"}
        assert entry["token"] == token["str"]
        assert entry["logprob_before"] == pytest.approx(token["logprob"], abs=1e-5)
        assert entry["delta"] == pytest.approx(entry["logprob_after"] - entry["logprob_before"], abs=1e-6)
    for index, token in enumerate(payload["tokens_after"]):
        assert token["i"] == index

    assert isinstance(payload["classification_note"], str)
    assert "removed" in payload["classification_note"]
    outcomes = payload["outcomes"]
    assert len(outcomes) == len(payload["deltas"])
    after_tokens = {token["str"] for token in payload["tokens_after"]}
    for entry, delta_entry in zip(outcomes, payload["deltas"]):
        assert set(entry) == {"token", "class", "delta", "in_caption_after"}
        assert entry["class"] in {"removed", "persisted", "changed"}
        assert entry["token"] == delta_entry["token"]
        assert entry["delta"] == pytest.approx(delta_entry["delta"], abs=1e-9)
        assert entry["in_caption_after"] == (entry["token"] in after_tokens)
        if entry["delta"] <= -1.0 and not entry["in_caption_after"]:
            expected = "removed"
        elif abs(entry["delta"]) < 0.2 and entry["in_caption_after"]:
            expected = "persisted"
        else:
            expected = "changed"
        assert entry["class"] == expected


def test_knockout_rejects_bad_patch(client: Any, session: dict[str, Any]) -> None:
    response = client.post("/api/knockout", json={"session_id": session["session_id"], "patches": [576]})
    assert response.status_code == 422
    assert "patch index" in response.json()["detail"]


# --------------------------------------------------------------------------- steering


def test_steer_noop_matches_baseline(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    token = generated["tokens"][0]["str"]
    job = _run_job(
        client,
        "/api/steer",
        {
            "session_id": session["session_id"],
            "layer": 0,
            "mode": "add",
            "token": token,
            "alpha": 0.0,
            "positions": "last",
            "max_new_tokens": GEN_TOKENS,
        },
    )
    assert job["status"] == "done", job
    payload = job["result"]
    assert payload["caption_before"] == generated["caption"]
    assert payload["caption_after"] == generated["caption"]
    assert payload["edits"] == [
        {
            "layer": 0,
            "mode": "add",
            "token": token,
            "source_token": None,
            "alpha": 0.0,
            "positions": "last",
        }
    ]
    assert payload["n_edit_forwards"] >= 0


def test_steer_alpha_shift(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    job = _run_job(
        client,
        "/api/steer",
        {
            "session_id": session["session_id"],
            "layer": 1,
            "mode": "add",
            "token": generated["tokens"][-1]["str"],
            "alpha": 8.0,
            "positions": "all",
            "max_new_tokens": GEN_TOKENS,
        },
    )
    assert job["status"] == "done", job
    payload = job["result"]
    assert isinstance(payload["caption_after"], str)
    assert len(payload["tokens_after"]) <= GEN_TOKENS
    assert payload["edits"][0]["positions"] == "all"
    assert payload["edits"][0]["alpha"] == 8.0


def test_steer_diagnostics(client: Any, session: dict[str, Any], generated: dict[str, Any]) -> None:
    add_job = _run_job(
        client,
        "/api/steer",
        {
            "session_id": session["session_id"],
            "layer": 0,
            "mode": "add",
            "token": generated["tokens"][0]["str"],
            "alpha": 1.0,
            "positions": "last",
            "max_new_tokens": 1,
        },
    )
    assert add_job["status"] == "done", add_job
    payload = add_job["result"]
    diagnostics = payload["diagnostics"]
    assert len(diagnostics) == len(payload["edits"]) == 1
    entry = diagnostics[0]
    assert set(entry) == {"layer", "mode", "v_norm", "h_norm", "cond"}
    assert entry["layer"] == 0 and entry["mode"] == "add"
    assert entry["v_norm"] > 0.0
    assert entry["cond"] is None  # only `swap` stacks two vectors
    assert entry["h_norm"] is None or entry["h_norm"] > 0.0

    # `swap` reports the condition number of the [v_s, v_t] basis, and `v_norm` is the
    # *target* token: two swaps sharing a target but not a source must agree on it.
    readout = _run_job(
        client,
        "/api/lens",
        {
            "session_id": session["session_id"],
            "layers": [0],
            "targets": [{"kind": "gen", "i": 0}],
            "topk": 3,
        },
    )
    assert readout["status"] == "done", readout
    top = [item["str"] for item in readout["result"]["targets"][0]["per_layer"][0]["topk"]]
    assert len(top) == 3 and len(set(top)) == 3
    norms = []
    for source in top[1:]:
        swap_job = _run_job(
            client,
            "/api/steer",
            {
                "session_id": session["session_id"],
                "layer": 0,
                "mode": "swap",
                "token": top[0],
                "source_token": source,
                "alpha": 1.0,
                "positions": "last",
                "max_new_tokens": 1,
            },
        )
        assert swap_job["status"] == "done", swap_job
        swap_entry = swap_job["result"]["diagnostics"][0]
        assert swap_entry["mode"] == "swap" and swap_entry["layer"] == 0
        assert swap_entry["v_norm"] > 0.0
        assert swap_entry["cond"] is None or (isinstance(swap_entry["cond"], float) and swap_entry["cond"] > 0.0)
        assert swap_entry["h_norm"] is None or swap_entry["h_norm"] > 0.0
        norms.append(swap_entry["v_norm"])
    assert norms[0] == pytest.approx(norms[1], rel=1e-6)


# --------------------------------------------------------------------------- twin


def test_twin_no_image_session(client: Any) -> None:
    """The ``no_image`` twin generates and steers, but has no patches."""
    created = client.post("/api/session", json={"variant": "no_image", "prompt": "Describe <image> this."})
    assert created.status_code == 200, created.text
    twin = created.json()
    assert twin["variant"] == "no_image"
    assert twin["image"] is None
    assert "<image>" not in twin["prompt_text"]
    assert twin["prompt_text"].replace("  ", " ") == "Describe this."
    assert twin["text_tokens"] and all(token["str"] != "<image>" for token in twin["text_tokens"])

    info = client.get(f"/api/session/{twin['session_id']}").json()
    assert info["variant"] == "no_image" and info["has_baseline"] is False

    generated = _run_job(client, "/api/generate", {"session_id": twin["session_id"], "max_new_tokens": 2})
    assert generated["status"] == "done", generated
    assert isinstance(generated["result"]["caption"], str)
    assert generated["result"]["tokens"], "the twin still generates without an image"

    lens_job = _run_job(
        client,
        "/api/lens",
        {"session_id": twin["session_id"], "targets": [{"kind": "prompt", "i": 0}], "topk": 2},
    )
    assert lens_job["status"] == "done", lens_job
    assert lens_job["result"]["targets"][0]["model_row"]["topk"]

    steer_job = _run_job(
        client,
        "/api/steer",
        {
            "session_id": twin["session_id"],
            "layer": 0,
            "mode": "add",
            "token": generated["result"]["tokens"][0]["str"],
            "alpha": 0.0,
            "positions": "last",
            "max_new_tokens": 1,
        },
    )
    assert steer_job["status"] == "done", steer_job
    assert len(steer_job["result"]["diagnostics"]) == 1

    for path, body in (
        (
            "/api/attribution",
            {"session_id": twin["session_id"], "layer": 0, "token": "a", "metric": "lens_prob"},
        ),
        ("/api/knockout", {"session_id": twin["session_id"], "patches": [0]}),
    ):
        blocked = client.post(path, json=body)
        assert blocked.status_code == 409, (path, blocked.status_code, blocked.text)
        assert "no_image" in blocked.json()["detail"]

    bad_variant = client.post("/api/session", json={"variant": "robot", "prompt": "x"})
    assert bad_variant.status_code == 422


# --------------------------------------------------------------------------- validation


def test_validation_errors(client: Any, session: dict[str, Any]) -> None:
    both = client.post("/api/session", json={"image_b64": "x", "image_path": "/tmp/x.png"})
    assert both.status_code == 422 and isinstance(both.json()["detail"], str)
    empty = client.post("/api/session", json={})
    assert empty.status_code == 422

    bad_generate = client.post("/api/generate", json={"session_id": session["session_id"], "max_new_tokens": 0})
    assert bad_generate.status_code == 422

    unknown_layer = client.post(
        "/api/lens",
        json={"session_id": session["session_id"], "layers": [9], "targets": [{"kind": "patch", "i": 0}]},
    )
    assert unknown_layer.status_code == 422
    bad_patch = client.post(
        "/api/lens",
        json={"session_id": session["session_id"], "targets": [{"kind": "patch", "i": 576}]},
    )
    assert bad_patch.status_code == 422
    bad_kind = client.post(
        "/api/lens",
        json={"session_id": session["session_id"], "targets": [{"kind": "vision", "i": 0}]},
    )
    assert bad_kind.status_code == 422

    bad_metric = client.post(
        "/api/attribution",
        json={"session_id": session["session_id"], "layer": 0, "token": "x", "metric": "nope"},
    )
    assert bad_metric.status_code == 422
    bad_token = client.post(
        "/api/attribution",
        json={"session_id": session["session_id"], "layer": 0, "token": ""},
    )
    assert bad_token.status_code == 422

    swap = client.post(
        "/api/steer",
        json={
            "session_id": session["session_id"],
            "layer": 0,
            "mode": "swap",
            "token": "x",
            "alpha": 1.0,
        },
    )
    assert swap.status_code == 422
    unfitted = client.post(
        "/api/steer",
        json={
            "session_id": session["session_id"],
            "layer": 2,
            "mode": "add",
            "token": "x",
            "alpha": 1.0,
        },
    )
    assert unfitted.status_code == 422
    missing_session = client.post("/api/generate", json={"session_id": "nope"})
    assert missing_session.status_code == 404


# --------------------------------------------------------------------------- no lens


def test_lens_unavailable_is_409(tmp_path_factory: pytest.TempPathFactory) -> None:
    empty_dir = tmp_path_factory.mktemp("vlmj-no-lens")
    with _build_client(tmp_path_factory.mktemp("vlmj-nolens"), lens_dir=empty_dir, autofit=False) as client:
        meta = client.get("/api/meta").json()
        assert meta["lens"]["available"] is False
        assert meta["lens"]["dir"] == str(empty_dir)
        assert isinstance(meta["lens"]["error"], str) and meta["lens"]["error"]

        missing = client.post("/api/session", json={"image_path": "missing.png"})
        assert missing.status_code == 422  # no such file, but the session path itself works
        created = client.post("/api/session", json={"image_b64": _sample_b64(client)})
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]

        for path, body in (
            ("/api/lens", {"session_id": session_id, "targets": [{"kind": "prompt", "i": 0}]}),
            (
                "/api/attribution",
                {"session_id": session_id, "layer": 0, "token": "x"},
            ),
            ("/api/knockout", {"session_id": session_id, "patches": [0]}),
            (
                "/api/steer",
                {"session_id": session_id, "layer": 0, "mode": "add", "token": "x", "alpha": 1.0},
            ),
        ):
            blocked = client.post(path, json=body)
            assert blocked.status_code == 409, (path, blocked.status_code, blocked.text)

        # Generation does not need the lens, so the demo stays usable without one.
        generated = _run_job(client, "/api/generate", {"session_id": session_id, "max_new_tokens": 2})
        assert generated["status"] == "done", generated
        assert isinstance(generated["result"]["caption"], str)


def _sample_b64(client: Any) -> str:
    return client.get("/api/samples").json()["samples"][0]["image_b64"]


# --------------------------------------------------------------------------- mock lens


def _load_mock_script() -> Any:
    spec = importlib.util.spec_from_file_location("vlmj_make_mock_lens", BACKEND_ROOT / "scripts" / "make_mock_lens.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mock_lens_script_roundtrip(tmp_path_factory: pytest.TempPathFactory) -> None:
    script = _load_mock_script()
    script._bootstrap()
    out_dir = tmp_path_factory.mktemp("vlmj-mock-lens")
    written = script.write_mock_lens("tiny", out_dir, seed=0)
    assert set(written) == {"text", "image", "all"}
    assert all(path.exists() for path in written.values())
    assert (out_dir / "provenance.json").exists()

    with _build_client(tmp_path_factory.mktemp("vlmj-mock-store"), lens_dir=out_dir, autofit=False) as client:
        meta = client.get("/api/meta").json()
        lens = meta["lens"]
        assert lens["available"] is True
        assert lens["mock"] is True
        assert lens["source_layers"] == [0, 1]
        assert lens["n_prompts"] == {"all": 0, "image": 0, "text": 0}
        assert "MOCK" in (lens["notes"] or "")

        created = client.post("/api/session", json={"image_b64": _sample_b64(client)})
        assert created.status_code == 200, created.text
        job = _run_job(
            client,
            "/api/lens",
            {
                "session_id": created.json()["session_id"],
                "targets": [{"kind": "prompt", "i": 0}],
                "topk": 2,
            },
        )
        assert job["status"] == "done", job
        assert job["result"]["targets"][0]["per_layer"]
