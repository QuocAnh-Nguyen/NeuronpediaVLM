#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build the three standard POPE splits (random / popular / adversarial) from a COCO
instances file, ready for the task-lens fit and the x24 yes/no microscope.

POPE asks yes/no object-presence questions per image: objects the image DOES contain
(label ``yes``) and objects it does NOT (label ``no``). The three splits differ only in
how the absent objects are chosen, which makes them a graded hallucination probe:

* ``random``      -- absent objects drawn uniformly from the non-present categories;
* ``popular``     -- absent objects taken from the dataset-frequency top list, walking
  it in order and skipping everything present on the image (a model that answers from
  global object priors instead of the image says ``yes`` here);
* ``adversarial`` -- absent objects ranked by co-occurrence with the image's present
  objects (image-level co-occurrence over the whole annotation file), so the prior of
  the present objects pulls the absent ones in.

Per image the builder emits up to ``QUESTIONS_PER_SIDE`` present-object and up to
``QUESTIONS_PER_SIDE`` absent-object questions; on full COCO val2014 (80 categories)
that is the standard 6 per image for any image with >= 3 annotated categories. Images
with fewer annotated categories contribute proportionally fewer questions on the short
side rather than repeating a category, and the counts land in each manifest's meta
(consumers compute yes-rate, they do not assume 0.5).

Outputs, per split, in ``--out-dir``: the raw record file ``<split>.jsonl`` (one
``{question_id, image, text, label}`` json object per line, exactly the schema
``vlm_lens.data.pope.read_pope_jsonl`` reads) plus the two prompt manifests built
through ``vlm_lens.data.pope.build_pope_manifest`` -- ``<split>-answered.jsonl``
(``include_answer=True``, the task-lens fit's teacher-forced answer positions) and
``<split>-open.jsonl`` (``include_answer=False``, what the microscope encodes so the
answer token never enters the prefill).

Deterministic: the image sample and the random-split draws are seeded per image
(``seed + split + file_name``), so the corpus is stable under ``--n-images`` changes;
popular/adversarial are rule-based and use no RNG at all.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.data.pope import build_pope_manifest  # noqa: E402

DEFAULT_ANNOTATIONS = "/data/baodq/coco2014/annotations/instances_val2014.json"
DEFAULT_IMAGES_DIR = "/data/baodq/coco2014/val2014"

#: The standard POPE question form (x18's QUESTION_TEMPLATE; data/pope.py wraps it
#: into ``USER: <image>\n{question}\nASSISTANT:``).
QUESTION_TEMPLATE = "Is there a {object} in the image?"

#: Questions per side (present / absent) per image, as in the official POPE files.
QUESTIONS_PER_SIDE = 3

#: The three standard splits, in output order.
SPLITS: tuple[str, ...] = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", default=DEFAULT_ANNOTATIONS,
        help="COCO instances json (categories/images/annotations)",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                        help="image directory written into the manifests' image paths")
    parser.add_argument("--n-images", type=int, default=500,
                        help="number of images to sample (seeded, with >= 1 annotated "
                             "category; images keep their questions independent of "
                             "later --n-images values)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True,
                        help="output directory for <split>.jsonl and the two "
                             "<split>-{answered,open}.jsonl manifests per split")
    return parser.parse_args()


# --------------------------------------------------------------------------- COCO


def load_coco(
    annotations_path: str,
) -> tuple[dict[int, str], list[tuple[str, set[int]]], dict[int, Counter[int]]]:
    """The COCO tables the builder needs, in one pass over the annotation file.

    Returns ``(name_by_id, per_image, cooccur)`` where ``per_image`` lists
    ``(file_name, present category ids)`` in file order (images with no annotations
    carry an empty set), and ``cooccur[a]`` is a ``Counter`` of the category ids that
    co-occur with ``a`` on an image, counted over ALL images (the dataset statistic the
    adversarial split ranks by).
    """
    with open(annotations_path, encoding="utf-8") as handle:
        data = json.load(handle)
    name_by_id = {int(cat["id"]): str(cat["name"]) for cat in data["categories"]}
    image_files = {int(img["id"]): str(img["file_name"]) for img in data["images"]}
    present: dict[int, set[int]] = defaultdict(set)
    for ann in data["annotations"]:
        present[int(ann["image_id"])].add(int(ann["category_id"]))
    per_image = [
        (image_files[image_id], present.get(image_id, set()))
        for image_id in image_files
    ]
    cooccur: dict[int, Counter[int]] = defaultdict(Counter)
    for categories in present.values():
        for a in categories:
            cooccur[a].update(b for b in categories if b != a)
    return name_by_id, per_image, cooccur


def select_images(
    per_image: list[tuple[str, set[int]]], n_images: int, seed: int
) -> list[tuple[str, set[int]]]:
    """The seeded image sample: the first ``n_images`` ANNOTATED images in a seeded
    shuffle (images with no annotations are skipped before sampling; nothing to ask
    about). The skip is visible in the printed summary's image counts.
    """
    annotated = [(name, cats) for name, cats in per_image if cats]
    rng = random.Random(f"pope-build:{seed}")
    order = list(annotated)
    rng.shuffle(order)
    return order[:n_images]


# ------------------------------------------------------------------ question plans


def absent_rank_popular(
    candidates: list[int], image_frequency: Counter[int]
) -> list[int]:
    """Popular-split absent objects: the dataset-frequency top list walked in order,
    skipping the image's own categories (see the module docstring)."""
    ranked = sorted(candidates, key=lambda c: (-image_frequency[c], c))
    return ranked


def absent_rank_adversarial(
    present: set[int], candidates: list[int],
    cooccur: dict[int, Counter[int]], image_frequency: Counter[int],
) -> list[int]:
    """Adversarial-split absent objects: highest co-occurrence with the image's present
    objects (sum over the present set), ties broken by dataset frequency then name id
    so the ranking is total and deterministic."""
    return sorted(
        candidates,
        key=lambda c: (
            -sum(cooccur[p][c] for p in present),
            -image_frequency[c],
            c,
        ),
    )


def question_plan(
    split: str,
    present_ids: set[int],
    name_by_id: dict[int, str],
    all_ids: list[int],
    image_frequency: Counter[int],
    cooccur: dict[int, Counter[int]],
    seed: int,
    file_name: str,
) -> list[tuple[str, bool]]:
    """One image's ``(question text, label)`` plan: up to 3 yes + up to 3 no.

    Present objects are a seeded sample of the image's annotated categories; the absent
    rule is split-specific (see the module docstring). The per-image seed makes every
    image's plan stable under ``--n-images`` changes (x18's convention).
    """
    rng = random.Random(f"pope:{seed}:{split}:{file_name}")
    present_names = sorted(name_by_id[c] for c in present_ids)
    n_present = min(QUESTIONS_PER_SIDE, len(present_names))
    plan: list[tuple[str, bool]] = [
        (QUESTION_TEMPLATE.format(object=name), True)
        for name in rng.sample(present_names, n_present)
    ]
    present_names_set = set(present_names)
    candidates = sorted(
        name_by_id[c] for c in all_ids if name_by_id[c] not in present_names_set
    )
    candidate_ids = [c for c in all_ids if c not in present_ids]
    if split == "random":
        n_absent = min(QUESTIONS_PER_SIDE, len(candidates))
        picked = rng.sample(candidates, n_absent)
    elif split == "popular":
        ranked_names = absent_rank_popular(candidate_ids, image_frequency)
        ranked = [name_by_id[c] for c in ranked_names]
        picked = ranked[:QUESTIONS_PER_SIDE]
    elif split == "adversarial":
        ranked_names = absent_rank_adversarial(
            present_ids, candidate_ids, cooccur, image_frequency
        )
        ranked = [name_by_id[c] for c in ranked_names]
        picked = ranked[:QUESTIONS_PER_SIDE]
    else:
        raise ValueError(f"unknown split {split!r} (expected one of {SPLITS})")
    plan.extend((QUESTION_TEMPLATE.format(object=name), False) for name in picked)
    return plan


# --------------------------------------------------------------------------- writer


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """One json object per line, via tmp + rename (the repo's atomic-write rule)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_split(
    split: str,
    images: list[tuple[str, set[int]]],
    name_by_id: dict[int, str],
    image_frequency: Counter[int],
    cooccur: dict[int, Counter[int]],
    seed: int,
    images_dir: str,
    out_dir: Path,
) -> dict[str, Any]:
    """One split end to end: records -> raw jsonl -> the two manifests. Returns the
    summary row for the printed table."""
    records: list[dict[str, Any]] = []
    for file_name, present_ids in images:
        plan = question_plan(
            split, present_ids, name_by_id, sorted(name_by_id), image_frequency,
            cooccur, seed, file_name,
        )
        for text, present in plan:
            records.append(
                {
                    "question_id": len(records),
                    "image": file_name,
                    "text": text,
                    "label": "yes" if present else "no",
                }
            )
    raw_path = out_dir / f"{split}.jsonl"
    write_jsonl_atomic(raw_path, records)
    answered = build_pope_manifest(
        out_dir / f"{split}-answered.jsonl",
        pope_jsonl=raw_path,
        images_dir=images_dir,
        include_answer=True,
    )
    opened = build_pope_manifest(
        out_dir / f"{split}-open.jsonl",
        pope_jsonl=raw_path,
        images_dir=images_dir,
        include_answer=False,
    )
    n_yes = sum(1 for record in records if record["label"] == "yes")
    return {
        "split": split,
        "raw": str(raw_path),
        "answered_manifest": str(answered),
        "open_manifest": str(opened),
        "n_questions": len(records),
        "n_yes": n_yes,
        "n_no": len(records) - n_yes,
        "n_images": len(images),
        "yes_rate": round(n_yes / len(records), 4) if records else None,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    name_by_id, per_image, cooccur = load_coco(args.annotations)
    image_frequency: Counter[int] = Counter()
    for _file_name, categories in per_image:
        image_frequency.update(categories)
    images = select_images(per_image, args.n_images, args.seed)
    if not images:
        raise ValueError(
            f"no annotated images found in {args.annotations}; nothing to build"
        )

    missing = sum(
        1 for file_name, _cats in images if not (Path(args.images_dir) / file_name).is_file()
    )
    if missing:
        print(
            f"note: {missing}/{len(images)} sampled images are missing under "
            f"{args.images_dir} (the manifests still name them; encoding them will fail)"
        )

    rows = [
        build_split(
            split, images, name_by_id, image_frequency, cooccur,
            args.seed, args.images_dir, out_dir,
        )
        for split in SPLITS
    ]

    n_unannotated = sum(1 for _f, cats in per_image if not cats)
    print(
        f"== build_pope: {len(images)} annotated images sampled (of {len(per_image)} "
        f"in the annotation file, {n_unannotated} unannotated and skipped; "
        f"seed={args.seed}) x {', '.join(SPLITS)} =="
    )
    print(
        f"{'split':<12}{'questions':>10}{'yes':>7}{'no':>7}{'yes_rate':>10}  files"
    )
    for row in rows:
        print(
            f"{row['split']:<12}{row['n_questions']:>10}{row['n_yes']:>7}"
            f"{row['n_no']:>7}{row['yes_rate']:>10}  {row['open_manifest']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
