#!/usr/bin/env python3
"""Globular clusters in a LIVE Milky Way: ProductionRunner(scheme=ClustersRunnerScheme(...)), one card.

The galaxy is an N-body realisation of the a4 'galaxy' model (tools/a4_galaxy_ics.py);
the clusters come from the catalogue (odisseo.globular_clusters) and are integrated by
nornax's hybrid Hermite scheme (internal direct sums, Hermite-4, n_sub fixed substeps),
coupled to the galaxy through jaccpot's FMM, with stripping (StrippingPolicy defaults).

Start points:
  --prepare DIR   the back-integrated centres of prepare.py (the run then ends near today)
  (default)       today's positions (the K ~ 150 bundle's functional run)

Outputs in --out (resumable: rerun the same command; it continues from restart.npz):
  rows.jsonl       one row per segment edge: the runner's diagnostics (moments, energy,
                   n_stripped, bound_mass, ledgers, energy_ledgered drift ...), retries,
                   re-plans, wall time and the device's memory stats
  snap_*.npz       every --snap-every edges: stripped stars and cluster members (+ a
                   galaxy surface-density map)
  run.json         the configuration; done.json at the end

usage (GPU, through mn-scratch/bin/on-free-card):
  python run_live.py --galaxy mw_1000000.npz --prepare runs/prepare --names "Pal 5" --M 4096 --out runs/pal5_live
  CPU small mode: add --small (galaxy 5000, M 64, 3 segments x 5 steps: tests the pipeline, not the physics)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

try:  # let py-spy attach (watchdogs read the stack before they kill)
    ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)
except Exception:  # noqa: BLE001
    pass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcmw  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--galaxy", required=True)
    p.add_argument("--n", type=int, default=None, help="galaxy subsample (default: the whole file)")
    p.add_argument("--catalogue", default=os.environ.get("ODISSEO_GC_DATA"))
    p.add_argument("--prepare", default=None, help="prepare.py's output dir (start at its back points)")
    p.add_argument("--names", nargs="+", default=None, help="clusters (default: prepare's; 'ALL': the catalogue)")
    p.add_argument("--max-clusters", type=int, default=None, help="with ALL: the K most massive")
    p.add_argument("--M", type=int, default=1000, help="stars per cluster")
    p.add_argument("--model", default="king", choices=("king", "plummer"))
    p.add_argument("--t-gyr", type=float, default=None, help="span (default: prepare's)")
    p.add_argument("--segments", type=int, default=None, help="stop after this many segments (functional runs)")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--n-sub", default="auto", help="internal substeps, or 'auto' (globular_clusters.default_n_sub)")
    p.add_argument("--steps", type=int, default=50, help="steps per segment")
    p.add_argument("--energy-every", type=int, default=1, help="segments between energy diagnostics")
    p.add_argument("--snap-every", type=int, default=10, help="segments between snapshots")
    p.add_argument("--restart-every", type=int, default=10)
    p.add_argument("--eps-g", type=float, default=None, help="galaxy softening (default 0.01 (N / 1e6)^(-1/3))")
    p.add_argument("--eps-c", type=float, default=None, help="cluster softening (default r_h,min / 26)")
    p.add_argument("--reserve", type=int, default=None)
    p.add_argument("--coupling", default="fmm", choices=("fmm", "direct", "none"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restore-preflight-patch", action=argparse.BooleanOptionalAction, default=True,
                   help="TEMPORARY jaccpot workaround when continuing a run (gcmw.patch_jaccpot_csr_restore_preflight)")
    p.add_argument("--small", action="store_true", help="CPU small mode: galaxy 5000, M 64, 3 segments x 5 steps")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.small:  # compile-dominated: ~4 min of compiles + ~2 min of steps on a free CPU host
        a.n = a.n or 5000
        a.M = min(a.M, 64)
        a.t_gyr = a.t_gyr or 0.002
        a.steps = min(a.steps, 5)
        a.snap_every = min(a.snap_every, 1)
        a.restart_every = min(a.restart_every, 2)
    return a


def main(argv=None):
    a = parse(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jaccpot.production import ProductionRunner, SegmentConfig
    from odisseo import globular_clusters as gc

    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    prep = None
    if a.prepare:
        with open(os.path.join(a.prepare, "prepare.json")) as fh:
            prep = json.load(fh)
    names = a.names or (prep["names"] if prep else ["Pal 5"])
    cat = gc.load_baumgardt(directory=a.catalogue)
    if names == ["ALL"]:
        rows = sorted((r for r in cat.values() if r.mass > 0 and r.r_half > 0), key=lambda r: -r.mass)
        names = [r.name for r in rows[: a.max_clusters or len(rows)]]
    centres = None
    if prep:
        if [gc.normalize_name(n) for n in names] != [gc.normalize_name(n) for n in prep["names"]]:
            raise SystemExit(f"--names {names} differ from prepare's {prep['names']}")
        centres = (np.array(prep["back"]["x"]), np.array(prep["back"]["v"]))
    t_gyr = a.t_gyr if a.t_gyr is not None else (prep["t_gyr"] if prep else 0.01)
    T = t_gyr * 1e3 / gcmw.TIME_MYR
    num_steps = int(round(T / a.dt / a.steps)) * a.steps
    if a.segments:
        num_steps = min(num_steps, a.segments * a.steps)

    xg, mg_all, meta = gcmw.load_galaxy(a.galaxy, a.n)
    N = len(mg_all)
    eps_g = a.eps_g or 0.01 * (N / 1e6) ** (-1.0 / 3.0)
    xc, vc, mc, table = gc.clusters_from_catalogue(names, a.M, model=a.model, catalogue=cat, centres=centres,
                                                   seed=a.seed, return_table=True)
    eps_c = a.eps_c or gc.default_cluster_softening(table)
    a.n_sub = gc.default_n_sub(a.dt, table) if a.n_sub == "auto" else int(a.n_sub)
    K = len(names)
    reserve = a.reserve if a.reserve is not None else (K * a.M if K == 1 else gc.default_reserve(K, a.M))
    gal, clu, settings = gc.build_solvers(eps_g, eps_c)
    clusters = gc.clusters_scheme((xc, vc, mc), eps_c=eps_c, cluster_solver=clu, n_sub=a.n_sub, reserve=reserve,
                                  coupling=a.coupling)
    config = dict(
        names=names, K=K, M=a.M, model=a.model, N_galaxy=N, galaxy=os.path.abspath(a.galaxy), galaxy_meta=meta,
        t_gyr=t_gyr, num_steps=num_steps, dt=a.dt, n_sub=a.n_sub, steps_per_segment=a.steps, eps_g=eps_g,
        eps_c=eps_c, reserve=reserve, coupling=a.coupling, backend=jax.default_backend(),
        device=str(jax.devices()[0]), settings=str(settings), prepare=a.prepare, seed=a.seed,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), xla_flags=os.environ.get("XLA_FLAGS"),
        allocator=os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
        jaccpot_env={k: v for k, v in os.environ.items() if k.startswith("JACCPOT_")},
        table=[dict(name=r["name"], mass_msun=r["mass_msun"], r_half_pc=r["r_half_pc"], w0=r["w0"]) for r in table],
    )
    gcmw.write_json(os.path.join(a.out, "run.json"), config)
    print(f"[run] K={K} M={a.M} N={N:,} eps_g={eps_g:.4g} eps_c={eps_c:.3g} n_sub={a.n_sub} t_cr_min={gc.crossing_times(table).min():.3g} reserve={reserve} steps={num_steps} "
          f"dt={a.dt} backend={jax.default_backend()}", flush=True)

    rows_path = os.path.join(a.out, "rows.jsonl")
    seg_wall = [time.time()]

    def mem():
        try:
            s = jax.devices()[0].memory_stats() or {}
            return dict(peak=s.get("peak_bytes_in_use"), now=s.get("bytes_in_use"))
        except Exception:  # noqa: BLE001
            return None

    def on_edge(e):
        d = dict(e.diagnostics or {})
        row = dict(segment=e.segment, step=e.step, time=e.time, retries=e.retries,
                   replans=(e.plan or {}).get("replans"), wall_s=e.wall_s, host_wall_s=time.time() - seg_wall[0],
                   mem=mem(), **{k: v for k, v in d.items()})
        seg_wall[0] = time.time()
        with open(rows_path, "a") as fh:
            fh.write(json.dumps(row, default=gcmw._jsonable) + "\n")
        last = e.step >= num_steps
        if e.carried is not None and (e.segment % a.snap_every == 0 or last):
            st = e.carried.state
            gid = np.asarray(st.galaxy_star_id)
            n_live = int(st.n_live)
            sel = np.flatnonzero(gid[:n_live] >= 0)
            gpos = np.asarray(st.galaxy_positions)
            hist_xy = np.histogram2d(gpos[:N, 0], gpos[:N, 1], bins=200, range=[[-3, 3], [-3, 3]])[0]
            hist_xz = np.histogram2d(gpos[:N, 0], gpos[:N, 2], bins=200, range=[[-3, 3], [-3, 3]])[0]
            cl = st.clusters
            np.savez_compressed(
                os.path.join(a.out, f"snap_{e.segment:05d}.npz"),
                time=e.time, step=e.step,
                stripped_x=gpos[sel], stripped_v=np.asarray(st.galaxy_velocities)[sel],
                stripped_m=np.asarray(st.galaxy_masses)[sel], stripped_id=gid[sel],
                cluster_x=np.asarray(cl.positions), cluster_v=np.asarray(cl.velocities),
                cluster_m=np.asarray(cl.masses), cluster_id=np.asarray(st.star_id),
                galaxy_xy=hist_xy.astype(np.float32), galaxy_xz=hist_xz.astype(np.float32),
            )
        print(f"[edge] seg {e.segment} step {e.step} t {e.time:.4f} n_stripped {d.get('n_stripped')} "
              f"dE_led {d.get('energy_ledgered_drift')} retries {e.retries} mem {row['mem']}", flush=True)

    restart = os.path.join(a.out, "restart.npz")
    seg = SegmentConfig(steps=a.steps, diagnostics_every=1, diagnostics_every_energy=a.energy_every,
                        restart_every=a.restart_every, restart_path=restart, domain_guard="off")
    runner = ProductionRunner(gal, settings=settings, segment=seg, scheme=clusters, callbacks=[on_edge])
    if os.path.exists(restart):
        patched = gcmw.patch_jaccpot_csr_restore_preflight() if a.restore_preflight_patch else False
        print(f"[run] continuing from {restart} (CSR restore preflight patch: {patched})", flush=True)
        res = runner.run(dt=a.dt, num_steps=num_steps, restart_from=restart)
    else:
        state = jnp.stack([jnp.asarray(xg[:, 0], jnp.float32), jnp.asarray(xg[:, 1], jnp.float32)], 1)
        res = runner.run(state, jnp.asarray(mg_all, jnp.float32), dt=a.dt, num_steps=num_steps)
    runner.save_restart(restart)  # the final edge: a rerun of the same command is a no-op
    done = dict(step=res.step, time=res.time, wall_s=time.time() - t_start, restored=res.restored,
                peak=mem(), segments=res.segment)
    gcmw.write_json(os.path.join(a.out, "done.json"), done)
    print("[done]", json.dumps(done, default=gcmw._jsonable), flush=True)


if __name__ == "__main__":
    main()
