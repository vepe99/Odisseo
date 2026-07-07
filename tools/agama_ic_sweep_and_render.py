#!/usr/bin/env python3
"""Sweep AGAMA IC parameters, score robustness, and render top candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import agama
import numpy as np
from autocvd import autocvd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=str, default="/tmp/agama_ic_sweep")
    p.add_argument("--n-particles", type=int, default=200000)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--top-k-render", type=int, default=2)
    p.add_argument("--disk-radial-grid", type=str, default="0.20,0.24,0.28")
    p.add_argument("--sigma-r0-grid", type=str, default="0.085,0.102,0.120")
    p.add_argument("--sigma-z0-grid", type=str, default="0.050,0.068,0.085")
    p.add_argument("--df-rsigmar", type=float, default=1.0)
    p.add_argument("--df-rsigmaz", type=float, default=1.0)
    p.add_argument("--reference-snapshots", type=str, default="/tmp/galaxy_strict_v2_200k_40_snapshots.npz")
    p.add_argument("--strict-exact-cap-match", action="store_true")
    return p.parse_args()


def _listf(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def score_snapshots(snapshot_npz: Path, reference_snapshot_npz: Path | None) -> dict[str, float]:
    z = np.load(snapshot_npz)
    sp = z["snapshot_positions"]  # [T, Ns, 3]
    rxy = np.linalg.norm(sp[:, :, :2], axis=2)
    zabs = np.abs(sp[:, :, 2])

    rmax_ref = float(np.percentile(rxy[0], 99.9))
    bins = np.linspace(0.0, max(rmax_ref, 1e-6), 161)
    centers = 0.5 * (bins[:-1] + bins[1:])
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    mask = (centers > 0.3) & (centers < 0.95 * rmax_ref)

    def frame_ring(rt: np.ndarray) -> tuple[float, float, float]:
        h, _ = np.histogram(rt, bins=bins)
        prof = h / np.maximum(area, 1e-12)
        smooth = np.convolve(prof, np.ones(9) / 9.0, mode="same")
        resid = (prof - smooth) / np.maximum(smooth, 1e-12)
        return (
            float(np.sqrt(np.mean(resid[mask] ** 2))),
            float(np.percentile(np.abs(resid[mask]), 95)),
            float(np.percentile(rt, 99)),
        )

    ring0, _, r99_0 = frame_ring(rxy[0])
    ringf, p95f, r99_f = frame_ring(rxy[-1])
    growth = float(r99_f / max(r99_0, 1e-12))
    thick0 = float(np.percentile(zabs[0], 90) / max(np.percentile(rxy[0], 90), 1e-12))
    thickf = float(np.percentile(zabs[-1], 90) / max(np.percentile(rxy[-1], 90), 1e-12))
    thick_growth = float(thickf / max(thick0, 1e-12))

    ref_penalty = 0.0
    if reference_snapshot_npz is not None and reference_snapshot_npz.exists():
        rz = np.load(reference_snapshot_npz)
        rr = np.linalg.norm(rz["snapshot_positions"][:, :, :2], axis=2)
        ref_r99_0 = float(np.percentile(rr[0], 99))
        ref_r50_0 = float(np.percentile(rr[0], 50))
        cand_r99_0 = float(np.percentile(rxy[0], 99))
        cand_r50_0 = float(np.percentile(rxy[0], 50))
        ref_penalty = abs(cand_r99_0 / max(ref_r99_0, 1e-12) - 1.0) + abs(
            cand_r50_0 / max(ref_r50_0, 1e-12) - 1.0
        )

    score = ringf + 1.5 * abs(growth - 1.0) + 0.7 * abs(thick_growth - 1.0) + 1.2 * ref_penalty
    return {
        "ring_rms_start": ring0,
        "ring_rms_end": ringf,
        "ring_p95_end": p95f,
        "r99_growth": growth,
        "thick_ratio_start": thick0,
        "thick_ratio_end": thickf,
        "thick_ratio_growth": thick_growth,
        "ref_profile_penalty": ref_penalty,
        "score": float(score),
    }


def _direct_self_accel_sample(
    pos: np.ndarray,
    mass: np.ndarray,
    target_idx: np.ndarray,
    *,
    softening_code: float = 0.002,
    source_chunk: int = 8192,
) -> np.ndarray:
    g = 1.0
    eps2 = float(softening_code) ** 2
    tgt = pos[target_idx]
    acc = np.zeros_like(tgt, dtype=np.float64)
    n = pos.shape[0]
    for s0 in range(0, n, int(source_chunk)):
        s1 = min(n, s0 + int(source_chunk))
        src = pos[s0:s1]
        msrc = mass[s0:s1]
        dr = src[None, :, :] - tgt[:, None, :]
        r2 = np.sum(dr * dr, axis=2) + eps2
        inv = 1.0 / np.sqrt(r2)
        inv3 = inv * inv * inv
        w = g * msrc[None, :] * inv3
        acc += np.einsum("ij,ijk->ik", w, dr, optimize=True)
    return acc


def _radial_component(pos: np.ndarray, acc: np.ndarray) -> np.ndarray:
    rxy = np.linalg.norm(pos[:, :2], axis=1)
    ex = np.where(rxy > 0, pos[:, 0] / np.maximum(rxy, 1e-12), 0.0)
    ey = np.where(rxy > 0, pos[:, 1] / np.maximum(rxy, 1e-12), 0.0)
    return acc[:, 0] * ex + acc[:, 1] * ey


def potential_consistency_metrics(ic_npz: Path, sample_targets: int = 256) -> dict[str, float]:
    z = np.load(ic_npz)
    state0 = np.asarray(z["state0"], dtype=np.float64)
    mass = np.asarray(z["mass"], dtype=np.float64)
    pos = state0[:, 0, :]
    n = pos.shape[0]
    k = min(int(sample_targets), n)
    idx = np.linspace(0, n - 1, k, dtype=np.int64)
    pos_t = pos[idx]

    # Reconstruct IC model potential from stored metadata.
    halo_mass = float(np.asarray(z["halo_mass_code"]))
    halo_rs = float(np.asarray(z["halo_rs_code"]))
    disk_mass = float(np.asarray(z["disk_mass_code"]))
    disk_rd = float(np.asarray(z["disk_radial_scale_code"]))
    disk_h = float(np.asarray(z["disk_height_code"]))

    agama.setUnits(length=1, mass=1, velocity=1)
    pot_ic = agama.Potential(
        dict(type="NFW", mass=halo_mass, scaleradius=halo_rs),
        dict(type="MiyamotoNagai", mass=disk_mass, scaleradius=disk_rd, scaleheight=max(1e-6, disk_h)),
    )
    pot_runtime_ext = agama.Potential(dict(type="NFW", mass=halo_mass, scaleradius=halo_rs))

    a_ic = np.asarray(pot_ic.force(pos_t), dtype=np.float64)
    a_ext = np.asarray(pot_runtime_ext.force(pos_t), dtype=np.float64)
    a_self = _direct_self_accel_sample(pos=pos, mass=mass, target_idx=idx, softening_code=0.002)
    a_runtime = a_ext + a_self

    ar_ic = _radial_component(pos_t, a_ic)
    ar_rt = _radial_component(pos_t, a_runtime)
    denom = np.maximum(np.abs(ar_rt), 1e-12)
    rel = np.abs(ar_ic - ar_rt) / denom
    return {
        "ar_mismatch_median": float(np.median(rel)),
        "ar_mismatch_p90": float(np.percentile(rel, 90)),
        "ar_mismatch_p99": float(np.percentile(rel, 99)),
    }


def rotation_curve_metrics(ic_npz: Path) -> dict[str, float]:
    z = np.load(ic_npz)
    state0 = np.asarray(z["state0"], dtype=np.float64)
    pos = state0[:, 0, :]
    vel = state0[:, 1, :]
    x, y = pos[:, 0], pos[:, 1]
    vx, vy = vel[:, 0], vel[:, 1]
    r = np.sqrt(x * x + y * y) + 1e-12
    vphi = (-y * vx + x * vy) / r
    vr = (x * vx + y * vy) / r

    # Mid-disk bins (avoid center/noisy outskirts)
    r_lo, r_hi = np.percentile(r, [15, 85])
    edges = np.linspace(r_lo, r_hi, 7)
    mids = 0.5 * (edges[:-1] + edges[1:])
    med = []
    prog = []
    sig = []
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if np.count_nonzero(m) < 128:
            continue
        vv = vphi[m]
        med.append(float(np.median(vv)))
        prog.append(float(np.mean(vv > 0)))
        sig.append(float(np.std(vv)))
    if len(med) < 3:
        return {
            "vphi_med_global": float(np.median(vphi)),
            "vphi_prograde_global": float(np.mean(vphi > 0)),
            "vphi_flatness": 1.0,
            "vphi_slope_abs": 1.0,
            "vr_dispersion_ratio": float(np.std(vr) / max(np.std(vphi), 1e-12)),
        }
    med = np.asarray(med)
    prog = np.asarray(prog)
    mids = mids[: len(med)]
    vmed_global = float(np.median(vphi))
    prograde_global = float(np.mean(vphi > 0))
    # Flatness: coefficient of variation of median vphi across bins
    flatness = float(np.std(med) / max(np.abs(np.mean(med)), 1e-12))
    # Rotation-curve slope (normalized)
    slope = np.polyfit(mids, med, 1)[0]
    slope_abs = float(np.abs(slope) / max(np.abs(np.mean(med)), 1e-12))
    vr_disp_ratio = float(np.std(vr) / max(np.std(vphi), 1e-12))
    return {
        "vphi_med_global": vmed_global,
        "vphi_prograde_global": prograde_global,
        "vphi_prograde_median_bin": float(np.median(prog)),
        "vphi_flatness": flatness,
        "vphi_slope_abs": slope_abs,
        "vr_dispersion_ratio": vr_disp_ratio,
    }


def main() -> None:
    args = parse_args()
    autocvd(num_gpus=1)

    repo = Path("/export/home/tbuck/Odisseo")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    gen_script = repo / "tools/agama_generate_scm_disk_ic.py"
    sim_script = repo / "notebooks/scalability/galaxy_disk_fmm_large_n.py"
    ref_snap = Path(args.reference_snapshots) if args.reference_snapshots else None

    env = os.environ.copy()
    env.update(
        {
            "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
            "JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH": "/tmp/jaccpot_static_strict_caps.json",
            "JACCPOT_STATIC_STRICT_CAP_RECORD": "0",
            "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "1"
            if args.strict_exact_cap_match
            else "0",
        }
    )

    rows: list[dict[str, float | str]] = []
    for rd in _listf(args.disk_radial_grid):
        for sr in _listf(args.sigma_r0_grid):
            for sz in _listf(args.sigma_z0_grid):
                tag = f"rd{rd:.3f}_sr{sr:.3f}_sz{sz:.3f}".replace(".", "p")
                print(f"[sweep] {tag}", flush=True)
                ic_path = workdir / f"ic_{tag}.npz"
                out_npz = workdir / f"run_{tag}.npz"
                snap_npz = workdir / f"snap_{tag}.npz"

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
                        "--rdisk-code",
                        str(rd),
                        "--qmin",
                        str(max(1.2, 1.0 + 3.0 * sr)),
                        "--rsigmar-code",
                        str(max(0.3, float(args.df_rsigmar) * rd)),
                    ],
                    check=True,
                    cwd=str(repo),
                    env=env,
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
                        str(out_npz),
                        "--save-snapshots",
                        "--snapshot-output",
                        str(snap_npz),
                        "--snapshot-stride",
                        "1",
                        "--snapshot-chunk-steps",
                        "1",
                    ],
                    check=True,
                    cwd=str(repo),
                    env=env,
                )

                metrics = score_snapshots(snap_npz, ref_snap)
                metrics.update(potential_consistency_metrics(ic_path))
                metrics.update(rotation_curve_metrics(ic_path))
                metrics["score"] = float(
                    metrics["score"]
                    + 1.0 * metrics["ar_mismatch_median"]
                    + 0.5 * metrics["ar_mismatch_p90"]
                    + 2.0 * max(0.0, 0.9 - metrics["vphi_prograde_global"])
                    + 1.0 * max(0.0, metrics["vphi_flatness"] - 0.25)
                    + 0.8 * max(0.0, metrics["vphi_slope_abs"] - 0.8)
                    + 0.5 * max(0.0, metrics["vr_dispersion_ratio"] - 1.0)
                )
                row = {
                    "tag": tag,
                    "disk_radial_scale_code": rd,
                    "sigma_r0_code": sr,
                    "sigma_z0_code": sz,
                    "ic_path": str(ic_path),
                    "run_path": str(out_npz),
                    "snap_path": str(snap_npz),
                    **metrics,
                }
                rows.append(row)
                print(f"[sweep] {tag} score={metrics['score']:.6f}", flush=True)

    rows.sort(key=lambda x: float(x["score"]))
    (workdir / "ranking.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (workdir / "ranking.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved rankings to {workdir / 'ranking.json'}", flush=True)

    for i, row in enumerate(rows[: max(0, int(args.top_k_render))], start=1):
        tag = str(row["tag"])
        ic_path = str(row["ic_path"])
        out_base = workdir / f"movie_top{i}_{tag}.mp4"
        print(f"[render-top] {tag}", flush=True)
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
                ic_path,
                "--no-ic-require-runtime-potential-match",
                "--output",
                str(workdir / f"movie_top{i}_{tag}.npz"),
                "--movie-path",
                str(out_base),
                "--movie-projections",
                "xy,xz",
                "--movie-fps",
                "20",
                "--render-backend",
                "density",
                "--render-resolution",
                "768",
                "--snapshot-stride",
                "1",
                "--snapshot-chunk-steps",
                "1",
            ],
            check=True,
            cwd=str(repo),
            env=env,
        )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
