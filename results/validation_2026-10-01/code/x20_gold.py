#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Gold/answer candidate resolution for X20, tokenizer-agnostic (the T29 fix).

WHY: x20's gold ids were llava-tokenizer-tuned — the first non-whitespace piece
of ``' ' + gold`` (plus order-ops' single-token synonyms). That convention misses
qwen2vl's byte-level-BPE surface forms and drove the qwen multihop pass@10 to
0.0: the multihop/multilingual prompts end with a quote or apostrophe
(``is "`` / ``es "`` / ``l'``), so the next emitted piece is the *bare* word form
(``"October`` -> ``October``), a form the convention never scores. This module
builds, per gold string, the candidate set the *_cand rank family scores:

1. the legacy convention ids (passed in by the caller; the legacy rank family
   keeps scoring exactly these, so llava numbers are unchanged),
2. all piece ids of the bare form ``tokenizer(gold, add_special_tokens=False)``
   (leading bos stripped — the tiny fixture's ``encode_text`` always prepends one),
3. the first-piece-of-each form: the first non-whitespace piece of the
   leading-space form (``vlm_registry.word_piece_ids``), i.e. the piece the model
   must emit first to start the word — for order-ops this also admits multi-piece
   synonym forms the single-token filter would have dropped,
4. decoded-match ids: every vocab id whose decoded surface, whitespace-stripped,
   equals the whitespace-stripped gold form (catches single-piece spellings with
   merged/odd spacing without per-id heuristics; strip-only normalization keeps
   case distinctions the data actually uses).

The rank metrics take the min over the candidate set (ties better) and pass@k is
"any candidate in top-k", so a candidate superset can only improve (never worsen)
the *_cand scores relative to scoring the legacy id alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

import vlm_registry as registry

__all__ = ["build_decode_index", "candidate_ids", "flat_ids"]

#: Ids decoded per ``batch_decode`` chunk when the tokenizer offers one (the
#: qwen vocab is ~152k ids; one call per id is needlessly slow).
_DECODE_CHUNK = 8192


def flat_ids(tokenizer: Any, text: str) -> list[int]:
    """Content ids of ``text``: flat ints, no specials, leading bos stripped.

    ``add_special_tokens=False`` never adds a bos on real tokenizers (whose
    content ids also never start with the bos id), while the tiny fixture's
    ``encode_text`` always prepends one — dropping a leading bos is safe for
    both (the x22 ``_content_ids`` convention). The tiny fixture returns a
    ``[1, k]`` tensor; real tokenizers return a list.
    """
    ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.reshape(-1).tolist()
    ids = [int(piece) for piece in ids]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and ids and ids[0] == int(bos):
        ids = ids[1:]
    return ids


def build_decode_index(tokenizer: Any) -> dict[str, list[int]]:
    """Normalized-decode surface -> vocab ids, for the decoded-match clause.

    One pass over the vocab (specials excluded): each id's decoded surface is
    whitespace-stripped and indexed. ``{}`` when the tokenizer exposes no vocab
    size (the clause is then skipped rather than guessed).
    """
    size = getattr(tokenizer, "vocab_size", None)
    if size is None and hasattr(tokenizer, "__len__"):
        size = len(tokenizer)
    if not size:
        return {}
    specials = set(getattr(tokenizer, "all_special_ids", None) or [])
    ids = [i for i in range(int(size)) if i not in specials]
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if batch_decode is not None:
        surfaces: list[str] = []
        for start in range(0, len(ids), _DECODE_CHUNK):
            # one single-id row per id: transformers 5 decodes a FLAT int list as
            # ONE sequence (concatenated), which would yield one surface per chunk
            surfaces.extend(batch_decode([[i] for i in ids[start:start + _DECODE_CHUNK]]))
    else:
        surfaces = [tokenizer.decode([i]) for i in ids]
    index: dict[str, list[int]] = {}
    for token_id, surface in zip(ids, surfaces):
        key = str(surface).strip()
        if key:
            index.setdefault(key, []).append(token_id)
    return index


def candidate_ids(
    model: Any,
    kind: str,
    forms: Sequence[str],
    legacy: Sequence[int],
    index: Mapping[str, list[int]] | None,
) -> list[int]:
    """The candidate set for one gold key: legacy ids ∪ the resolver's additions.

    ``forms`` is the key's surface-form set (order-ops synonym expansion done by
    the caller; a plain suite passes ``(key,)``). Per form the additions are the
    bare-form pieces (all of them), the first-piece-of-each leading-space form,
    and the decoded-match ids from ``index`` (``None`` skips that clause).
    Deduplicated preserving insertion order — legacy ids first, so the candidate
    set is a strict superset of what the legacy family scores.
    """
    tokenizer = model.tokenizer
    out = list(dict.fromkeys(int(token_id) for token_id in legacy))
    for form in forms:
        out.extend(flat_ids(tokenizer, form))
        pieces = registry.word_piece_ids(model, kind, form)
        if pieces:
            out.append(int(pieces[0]))
        if index is not None:
            out.extend(index.get(form.strip(), ()))
    return list(dict.fromkeys(out))
