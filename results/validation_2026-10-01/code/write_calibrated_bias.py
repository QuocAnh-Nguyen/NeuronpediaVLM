#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Emit a calibrated bias dir: apply the calib_readout.py winners to a census bias file.

``calib_readout.py`` picks a per-layer scale multiplier and output temperature by
minimizing KL(lens softmax || model softmax) on a calibration split; this script bakes
the winners into the ``bias-<mask>.pt`` family the standard scorer consumes. Per layer
with a calib entry:

    scale <- scale * best_scale_mult,   temp <- best_temp   (omitted when 1.0),

so the readout becomes ``unembed(s_l * m_l * (J_l @ h + b_l)) / T_l`` - the payload
contract's ``"temp"`` key (absent => 1.0, so a winner of 1.0 leaves the file's
temperature behavior bit-identical to the census default). Layers of the base file
missing from the calib summary are carried through unchanged (warned); the census meta
is preserved and annotated with the calibration provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlm_lens.artifacts import load_bias, save_bias  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-bias-dir", required=True,
        help="directory of moment-census bias-<mask>.pt affine-correction files",
    )
    parser.add_argument("--calib", default="calib.json", help="the calib_readout.py summary (per-layer winners)")
    parser.add_argument("--out", required=True, help="the calibrated bias directory to write")
    parser.add_argument("--mask", default="text", help="which bias-<mask>.pt file to calibrate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    calib_layers: dict[str, dict] = report.get("layers", {})
    bias_path = Path(args.base_bias_dir) / f"bias-{args.mask}.pt"
    if not bias_path.is_file():
        raise FileNotFoundError(f"no bias-{args.mask}.pt under {args.base_bias_dir}")
    bias_payload, meta = load_bias(bias_path)

    print(f"calibrating {bias_path} with {args.calib} (mask {args.mask!r})")
    header = f"{'layer':<6}  {'old scale':>10}  {'new scale':>10}  {'temp':>7}   kl_base -> kl_best"
    print(header)
    print("-" * len(header))
    new_payload: dict[str, dict] = {}
    n_calibrated = 0
    for layer_str, entry in bias_payload.items():
        row = calib_layers.get(layer_str)
        old_scale = float(entry["scale"])
        new_entry = dict(entry)
        if row is None:
            print(f"{layer_str:<6}  {old_scale:>10.3f}  {old_scale:>10.3f}  {'-':>7}   -")
            new_payload[layer_str] = new_entry
            continue
        best_scale_mult = float(row["best_scale_mult"])
        best_temp = float(row["best_temp"])
        new_scale = old_scale * best_scale_mult
        new_entry["scale"] = new_scale
        # The payload contract: "temp" present => divide by temp; absent => 1.0. A stale
        # temp from a previous calibration must go when the new winner is 1.0.
        if best_temp == 1.0:
            new_entry.pop("temp", None)
        else:
            new_entry["temp"] = best_temp
        new_payload[layer_str] = new_entry
        n_calibrated += 1
        print(
            f"{layer_str:<6}  {old_scale:>10.3f}  {new_scale:>10.3f}  {best_temp:>7.2f}   "
            f"{row['kl_base']:.3f} -> {row['kl_best']:.3f}"
        )
    if not n_calibrated:
        print(f"WARNING: no layers of {bias_path.name} appear in {args.calib}; copied unchanged")
    if n_calibrated < len(calib_layers):
        missing = sorted(set(calib_layers) - set(bias_payload))
        print(f"WARNING: {missing} in the calib summary but not in {bias_path.name}; skipped")

    new_meta = {
        **meta,
        "calibrated_from": str(bias_path),
        "calib_json": str(args.calib),
        "mask": args.mask,
        "n_layers_calibrated": n_calibrated,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = save_bias(
        Path(args.out) / f"bias-{args.mask}.pt", {**new_payload, "meta": new_meta}
    )
    print(f"\ncalibrated {n_calibrated}/{len(bias_payload)} layers")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
