# SPDX-License-Identifier: Apache-2.0
"""Per-model dispatch shared by the fit and equivalence CLIs.

One :class:`ModelSpec` per fittable checkpoint family ties together everything the
scripts need: the ``LensModel`` wrapper class, the HuggingFace checkpoint id, the tiny
CPU fixture builder, the image-token id the checkpoint expands image placeholders to,
and the default artifacts subdirectory under :data:`FITS_ROOT`.

Wrapper and fixture modules are referenced by dotted path and imported lazily, so
fitting ``llava-7b`` never imports the qwen2-vl stack (and vice versa) and the table
itself stays importable while a backend is still under construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

#: Root directory for fitted artifacts when a script's ``--out`` is not given.
FITS_ROOT = Path("fits")


def _resolve(dotted: str) -> Any:
    module_name, _, attr = dotted.rpartition(".")
    return getattr(import_module(module_name), attr)


@dataclass(frozen=True)
class ModelSpec:
    """Identity of one checkpoint family: wrapper, checkpoint, fixture, artifact layout.

    ``image_token_id`` describes the *real* checkpoint; tiny fixtures seed their own
    vocabulary and are free to use a different placeholder id.
    """

    key: str
    hf_id: str
    lens_model: str
    tiny_builder: str
    tiny_config: str | None
    tiny_id: str
    out_subdir: str
    image_token_id: int

    def lens_model_cls(self) -> Any:
        """The ``LensModel`` wrapper class, imported on first use."""
        return _resolve(self.lens_model)

    def build_tiny(self, *, image_size: int | None = None) -> tuple[Any, Any]:
        """Random-weight CPU fixture as ``(hf_model, processor)``.

        ``image_size`` is forwarded to the fixture config when the builder takes one;
        builders without a config class (``tiny_config=None``) are built with defaults.
        """
        builder = _resolve(self.tiny_builder)
        if self.tiny_config is None:
            return builder(None)
        config_cls = _resolve(self.tiny_config)
        config = config_cls(image_size=image_size) if image_size is not None else config_cls()
        return builder(config)


MODEL_SPECS: dict[str, ModelSpec] = {
    "llava-7b": ModelSpec(
        key="llava-7b",
        hf_id="llava-hf/llava-1.5-7b-hf",
        lens_model="vlm_lens.models.llava.LlavaLensModel",
        tiny_builder="vlm_lens.models.tiny_llava.build_tiny_llava",
        tiny_config="vlm_lens.models.tiny_llava.TinyLlavaConfig",
        tiny_id="tiny-llava",
        out_subdir="llava-1.5-7b-hf",
        image_token_id=32000,
    ),
    "llava-13b": ModelSpec(
        key="llava-13b",
        hf_id="llava-hf/llava-1.5-13b-hf",
        lens_model="vlm_lens.models.llava.LlavaLensModel",
        tiny_builder="vlm_lens.models.tiny_llava.build_tiny_llava",
        tiny_config="vlm_lens.models.tiny_llava.TinyLlavaConfig",
        tiny_id="tiny-llava",
        out_subdir="llava-1.5-13b-hf",
        image_token_id=32000,
    ),
    "qwen2vl": ModelSpec(
        key="qwen2vl",
        hf_id="Qwen/Qwen2-VL-7B-Instruct",
        lens_model="vlm_lens.models.qwen2vl.Qwen2VLLensModel",
        tiny_builder="vlm_lens.models.tiny_qwen2vl.build_tiny_qwen2vl",
        tiny_config=None,
        tiny_id="tiny-qwen2vl",
        out_subdir="qwen2-vl-7b-instruct",
        image_token_id=151655,
    ),
}

__all__ = ["FITS_ROOT", "MODEL_SPECS", "ModelSpec"]
