#!/usr/bin/env python3
"""Generate concise bottleneck tables from fused audit summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _render_markdown(summary: dict[str, Any], *, top_n: int) -> str:
    rows = list(summary.get("bottleneck_table", []))[: max(1, int(top_n))]
    lines = [
        "# Fused Audit Bottleneck Table",
        "",
        f"- Audit tag: `{summary.get('audit', {}).get('audit_tag')}`",
        f"- Run class: `{summary.get('audit', {}).get('run_class')}`",
        f"- Delta variant-baseline seconds: `{summary.get('delta_variant_minus_baseline_seconds')}`",
        "",
        "## Gate Flags",
        "",
    ]
    for key, val in summary.get("gate_flags", {}).items():
        if key == "gate_context":
            continue
        lines.append(f"- {key}: `{val}`")
    lines += [
        "",
        "## Top Buckets",
        "",
        "| Bucket | Timeline Region | Baseline s | Variant s | Delta v-b s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('bucket')} | {row.get('timeline_region')} | "
            f"{float(row.get('baseline_seconds', 0.0)):.6f} | "
            f"{float(row.get('variant_seconds', 0.0)):.6f} | "
            f"{float(row.get('delta_variant_minus_baseline_seconds', 0.0)):.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True, help="Path to audit_summary.json")
    p.add_argument("--top-n", type=int, default=12, help="Top-N buckets to include")
    p.add_argument("--output-md", type=Path, default=None, help="Optional output markdown path")
    p.add_argument("--output-json", type=Path, default=None, help="Optional filtered JSON output path")
    args = p.parse_args()

    with args.summary.open(encoding="utf-8") as f:
        summary = json.load(f)

    md = _render_markdown(summary, top_n=int(args.top_n))
    print(md)

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")

    if args.output_json is not None:
        payload = {
            "audit": summary.get("audit", {}),
            "delta_variant_minus_baseline_seconds": summary.get(
                "delta_variant_minus_baseline_seconds"
            ),
            "gate_flags": summary.get("gate_flags", {}),
            "bottleneck_table": list(summary.get("bottleneck_table", []))[
                : max(1, int(args.top_n))
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
