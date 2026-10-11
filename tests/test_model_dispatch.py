# SPDX-License-Identifier: Apache-2.0
"""``--model`` dispatch: registry contents, CLI defaults, and tiny-backend gates.

The default path (``--model llava-7b``) must stay identical to the pre-dispatch
behaviour, so the tests pin the resolved ids and run both CLIs end-to-end on the tiny
CPU fixtures. The qwen2-vl tests skip (with the reason recorded) while the wrapper or
fixture module is absent or still under construction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from vlm_lens.models.registry import FITS_ROOT, MODEL_SPECS

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Import a script module by path (``scripts/`` is not a package)."""
    location = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests_{name}", location)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit_llava = _load_script("fit_llava")
check_equivalence = _load_script("check_equivalence")


def _qwen2vl_fixture_error() -> str | None:
    """Why the tiny qwen2-vl fixture is unusable, or ``None`` when importable."""
    try:
        import vlm_lens.models.tiny_qwen2vl  # noqa: F401
    except Exception as error:  # fixture may still be under construction; guarded
        return f"{type(error).__name__}: {error}"
    return None


QWEN2VL_SKIP = _qwen2vl_fixture_error()
qwen2vl_ready = pytest.mark.skipif(
    QWEN2VL_SKIP is not None,
    reason=f"tiny qwen2vl fixture unavailable: {QWEN2VL_SKIP}",
)


def test_dispatch_table_maps_model_families():
    assert set(MODEL_SPECS) == {"llava-7b", "llava-13b", "qwen2vl"}
    seven_b = MODEL_SPECS["llava-7b"]
    thirteen_b = MODEL_SPECS["llava-13b"]
    qwen = MODEL_SPECS["qwen2vl"]
    assert (seven_b.hf_id, seven_b.image_token_id) == ("llava-hf/llava-1.5-7b-hf", 32000)
    assert (thirteen_b.hf_id, thirteen_b.image_token_id) == ("llava-hf/llava-1.5-13b-hf", 32000)
    assert (qwen.hf_id, qwen.image_token_id) == ("Qwen/Qwen2-VL-7B-Instruct", 151655)
    assert seven_b.lens_model_cls().__name__ == "LlavaLensModel"
    assert seven_b.lens_model == thirteen_b.lens_model
    # wrappers are referenced lazily: the table names, not imports, the qwen2-vl stack
    assert qwen.lens_model.endswith("Qwen2VLLensModel")
    assert qwen.tiny_builder.endswith("build_tiny_qwen2vl")
    # artifacts layout: one subdir per checkpoint family under the fits root
    assert seven_b.out_subdir == "llava-1.5-7b-hf"
    assert thirteen_b.out_subdir == "llava-1.5-13b-hf"
    assert qwen.out_subdir == "qwen2-vl-7b-instruct"


def test_fit_llava_defaults_match_pre_dispatch_behavior():
    args = fit_llava.parse_args([])
    assert args.backend == "hf-llava"
    assert args.model == "llava-7b"
    assert MODEL_SPECS[args.model].hf_id == "llava-hf/llava-1.5-7b-hf"
    # explicit --out wins; omission falls back to fits/<model subdir>
    assert fit_llava.resolve_out_dir(fit_llava.parse_args(["--out", "runs/x"])) == Path("runs/x")
    assert fit_llava.resolve_out_dir(args) == FITS_ROOT / "llava-1.5-7b-hf"
    assert fit_llava.resolve_out_dir(fit_llava.parse_args(["--model", "qwen2vl"])) == (
        FITS_ROOT / "qwen2-vl-7b-instruct"
    )


def test_fit_llava_tiny_backend_builds_default_wrapper():
    model = fit_llava.build_model(
        fit_llava.parse_args(["--backend", "tiny", "--tiny-image-size", "56"])
    )
    assert type(model).__name__ == "LlavaLensModel"
    assert model.n_layers == 4  # TinyLlavaConfig defaults
    assert model.d_model == 32
    # fixtures seed their own vocabulary: the placeholder id is the fixture's, not 32000
    assert model.image_token_id == 100


def test_fit_llava_tiny_end_to_end_writes_provenance(tmp_path):
    out = tmp_path / "fit"
    rc = fit_llava.main(
        [
            "--backend",
            "tiny",
            "--corpus",
            "dummy",
            "--n-samples",
            "2",
            "--tiny-image-size",
            "56",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    provenance = json.loads((out / "artifacts" / "provenance.json").read_text())
    assert provenance["extra"]["model_id"] == "tiny-llava"
    assert provenance["extra"]["lens_model_class"] == "LlavaLensModel"
    assert provenance["model"]["class"] == "LlavaForConditionalGeneration"
    assert (out / "artifacts" / "lens-text.pt").exists()


def test_check_equivalence_default_gate_passes(capsys):
    assert check_equivalence.main(["--backend", "tiny"]) == 0
    assert "EQUIVALENCE PASS" in capsys.readouterr().out


@qwen2vl_ready
def test_check_equivalence_qwen2vl_tiny_gate_passes(capsys):
    # exercises the dict-returning forward_residual through the gate's normalization
    assert check_equivalence.main(["--backend", "tiny", "--model", "qwen2vl"]) == 0
    assert "EQUIVALENCE PASS" in capsys.readouterr().out


@qwen2vl_ready
def test_fit_llava_qwen2vl_tiny_end_to_end_writes_provenance(tmp_path):
    out = tmp_path / "fit"
    rc = fit_llava.main(
        [
            "--backend",
            "tiny",
            "--model",
            "qwen2vl",
            "--corpus",
            "dummy",
            "--n-samples",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    provenance = json.loads((out / "artifacts" / "provenance.json").read_text())
    assert provenance["extra"]["model_id"] == "tiny-qwen2vl"
    assert provenance["extra"]["lens_model_class"] == "Qwen2VLLensModel"
    assert provenance["model"]["class"] == "Qwen2VLForConditionalGeneration"
    assert (out / "artifacts" / "lens-text.pt").exists()
