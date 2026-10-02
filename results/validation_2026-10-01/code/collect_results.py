#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Collect every campaign artifact into one raw index, one flat CSV and the exact commands.

The report quotes *numbers*; this script makes them traceable. It walks the run directory,
hashes and summarises every JSON/JSONL artifact, flattens all scalar leaves of the analysis
reports into a single long-format CSV (file, json path, value) without interpreting them, and
copies out the invocation lines of the step scripts plus their hashes. No judgement, no
metrics - anything computed here can be recomputed from the JSONs it points at.

Usage (from the repo root):
    python code/collect_results.py --run-dir /data/vlm-lens/validation \
        --code-dir code --out-dir . --label validation_2026-10-01

Writes ``composition.json``, ``results.csv``, ``commands.txt`` next to ``--out-dir``. Reruns
are idempotent; missing steps are listed, not invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

__all__ = ["main", "walk_leaves", "sha256_file"]

#: Text files hashed in full; anything larger is hashed in chunks (the .pt lenses never are).
CHUNK = 1 << 20
#: Depth guard for pathological nesting; all campaign JSONs stay far below it.
MAX_DEPTH = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def walk_leaves(value: Any, prefix: str = "", depth: int = 0) -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted.path, scalar)`` for every scalar leaf; lists index with ``[i]``."""
    if depth > MAX_DEPTH:
        yield prefix or "<root>", "<truncated>"
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_leaves(item, child, depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_leaves(item, f"{prefix}[{index}]", depth + 1)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield prefix or "<root>", value
    else:  # pragma: no cover - JSON cannot produce this
        yield prefix or "<root>", repr(value)


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def collect_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    """Every JSON/JSONL artifact under ``run_dir``, hashed, with its parse status."""
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            continue
        record: dict[str, Any] = {
            "path": str(path.relative_to(run_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        try:
            payload: Any = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(
                path.read_text(encoding="utf-8")
            )
            record["parse"] = "ok"
            record["top_level"] = (
                sorted(payload)[:24] if isinstance(payload, dict) else f"list[{len(payload)}]"
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:  # truncated mid-write
            record["parse"] = f"error: {error}"
        artifacts.append(record)
    return artifacts


def flatten_reports(run_dir: Path, out_csv: Path) -> int:
    """Long-format CSV of every scalar leaf in every parseable JSON/JSONL artifact."""
    rows_written = 0
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "json_path", "value"])
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.suffix not in (".json", ".jsonl"):
                continue
            try:
                payload: Any = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(
                    path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            relative = str(path.relative_to(run_dir))
            items = payload if isinstance(payload, list) else [payload]
            for index, item in enumerate(items):
                root = f"[{index}]" if isinstance(payload, list) else ""
                for json_path, value in walk_leaves(item, root):
                    writer.writerow([relative, json_path, json.dumps(value)])
                    rows_written += 1
    return rows_written


def step_commands(code_dir: Path) -> list[str]:
    """Invocation lines of the step scripts, verbatim, in file order."""
    lines: list[str] = []
    for script in sorted(code_dir.glob("*.sh")):
        for raw in script.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", stripped):
                continue  # a variable assignment, not an invocation
            if re.search(r'("\$P"|\$PY|python[0-9.]*\s|\.py\b)', stripped):
                lines.append(f"{script.name}: {stripped}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="campaign output tree to walk")
    parser.add_argument("--code-dir", default="code", help="directory with run_*.sh step scripts")
    parser.add_argument("--out-dir", default=".", help="where composition.json/results.csv land")
    parser.add_argument("--label", default="campaign", help="label embedded in the index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    code_dir = Path(args.code_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not run_dir.is_dir():
        print(f"run dir missing: {run_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = collect_artifacts(run_dir)
    csv_path = out_dir / "results.csv"
    rows_written = flatten_reports(run_dir, csv_path)

    code_files = [
        {"path": script.name, "sha256": sha256_file(script)}
        for script in sorted(code_dir.glob("*.sh")) + sorted(code_dir.glob("*.py"))
    ]
    composition = {
        "label": args.label,
        "run_dir": str(run_dir),
        "code_dir": str(code_dir),
        "n_artifacts": len(artifacts),
        "csv_rows": rows_written,
        "artifacts": artifacts,
        "code": code_files,
    }
    (out_dir / "composition.json").write_text(
        json.dumps(composition, indent=2), encoding="utf-8"
    )
    (out_dir / "commands.txt").write_text(
        "\n".join(step_commands(code_dir)) + "\n", encoding="utf-8"
    )

    parsed = sum(1 for record in artifacts if record["parse"] == "ok")
    print(f"artifacts: {len(artifacts)} ({parsed} parseable), csv rows: {rows_written}")
    print(f"code files hashed: {len(code_files)}")
    print(f"wrote {out_dir / 'composition.json'}, {csv_path}, {out_dir / 'commands.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
