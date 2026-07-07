#!/usr/bin/env python3
"""Sweep galpy DF IC parameters and score ringiness/drift using short Odisseo runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from autocvd import autocvd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=str, default="/tmp/galpy_ic_sweep")
    p.add_argument("--n-particles", type=int, default=200_000)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--sigma-r-grid", type=str, default="0.10,0.14,0.18")
    p.add_argument("--sigma-z-grid-kms", type=str, default="2.0,4.0,6.0")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def ring_score(snapshot_npz: Path) -> dict[str, float]:
    z = np.load(snapshot_npz)
    sp = z["snapshot_positions"]
    r = np.linalg.norm(sp[:, :, :2], axis=2)
    rmax_ref = float(np.percentile(r[0], 99.9))
    bins = np.linspace(0.0, max(rmax_ref, 1e-6), 161)
    centers = 0.5 * (bins[:-1] + bins[1:])
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    mask = (centers > 0.3) & (centers < 0.95 * rmax_ref)

    def one_frame(rt: np.ndarray) -> tuple[float, float, float]:
        h, _ = np.histogram(rt, bins=bins)
        prof = h / np.maximum(area, 1e-12)
        smooth = np.convolve(prof, np.ones(9) / 9.0, mode="same")
        resid = (prof - smooth) / np.maximum(smooth, 1e-12)
        return (
            float(np.sqrt(np.mean(resid[mask] ** 2))),
            float(np.percentile(np.abs(resid[mask]), 95)),
            float(np.percentile(rt, 99)),
        )

    rms0, p950, r990 = one_frame(r[0])
    rmsf, p95f, r99f = one_frame(r[-1])
    drift = float(r99f / max(r990, 1e-12))
    return {
        "ring_rms_start": rms0,
        "ring_rms_end": rmsf,
        "ring_p95_start": p950,
        "ring_p95_end": p95f,
        "r99_growth": drift,
        "score": rmsf + 2.0 * abs(drift - 1.0),
    }


def main() -> None:
    args = parse_args()
    autocvd(num_gpus=1)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    repo = Path("/export/home/tbuck/Odisseo")
    gen_script = repo / "tools/galpy_generate_df_ic.py"
    sim_script = repo / "notebooks/scalability/galaxy_disk_fmm_large_n.py"
    sig_r = [float(x) for x in str(args.sigma_r_grid).split(",") if x.strip()]
    sig_z = [float(x) for x in str(args.sigma_z_grid_kms).split(",") if x.strip()]

    env = os.environ.copy()
    env.update(
        {
            "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
            "JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH": "/tmp/jaccpot_static_strict_caps.json",
            "JACCPOT_STATIC_STRICT_CAP_RECORD": "0",
            "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "1",
        }
    )

    rows: list[dict[str, float | str]] = []
    for sr in sig_r:
        for sz in sig_z:
            tag = f"sr{sr:.3f}_sz{sz:.1f}".replace(".", "p")
            ic_path = workdir / f"ic_{tag}.npz"
            out_path = workdir / f"run_{tag}.npz"
            snap_path = workdir / f"snap_{tag}.npz"
            print(f"[sweep] candidate {tag}")
            subprocess.run(
                [
                    "micromamba",
                    "run",
                    "-n",
                    "odisseo",
                    "python",
                    str(gen_script),
                    "--output",
                    str(ic_path),
                    "--n-particles",
                    str(int(args.n_particles)),
                    "--seed",
                    str(int(args.seed)),
                    "--sigma-r-over-vc",
                    str(sr),
                    "--sigma-z-kms",
                    str(sz),
                ],
                check=True,
                env=env,
                cwd=str(repo),
            )
            subprocess.run(
                [
                    "micromamba",
                    "run",
                    "-n",
                    "odisseo",
                    "python",
                    str(sim_script),
                    "--mode",
                    "render",
                    "--n-particles",
                    str(int(args.n_particles)),
                    "--num-steps",
                    str(int(args.num_steps)),
                    "--fmm-preset",
                    "large_n_gpu",
                    "--fmm-runtime-path",
                    "large_n",
                    "--fmm-tree-build-mode",
                    "static_radix",
                    "--fmm-leaf-size",
                    str(int(args.leaf_size)),
                    "--fmm-refresh-every",
                    "1",
                    "--no-fmm-large-n-environment-overrides",
                    "--ic-source",
                    "load",
                    "--ic-input-path",
                    str(ic_path),
                    "--no-ic-require-runtime-potential-match",
                    "--output",
                    str(out_path),
                    "--save-snapshots",
                    "--snapshot-output",
                    str(snap_path),
                    "--snapshot-stride",
                    "1",
                    "--snapshot-chunk-steps",
                    "1",
                ],
                check=True,
                env=env,
                cwd=str(repo),
            )
            metrics = ring_score(snap_path)
            row = {
                "tag": tag,
                "sigma_r_over_vc": sr,
                "sigma_z_kms": sz,
                **metrics,
                "ic_path": str(ic_path),
            }
            rows.append(row)
            print(f"[sweep] {tag} score={metrics['score']:.5f}")

    rows.sort(key=lambda r: float(r["score"]))
    out_json = workdir / "ranking.json"
    out_csv = workdir / "ranking.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved ranking: {out_json}")
    print(f"Saved ranking: {out_csv}")
    for i, r in enumerate(rows[: max(1, int(args.top_k))], start=1):
        print(f"TOP{i}: {r['tag']} score={float(r['score']):.5f} ic={r['ic_path']}")


if __name__ == "__main__":
    main()
