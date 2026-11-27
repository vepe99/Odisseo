#!/usr/bin/env python3
"""The sanity check's other arm: the same clusters as plain N-body systems in Odisseo's STATIC MW.

The stars are the live run's (same catalogue rows, model, M and seed: the same
realisation, at prepare.py's back points). Odisseo's own direct-sum leapfrog
(``time_integration``, DIRECT_ACC_MATRIX, Plummer softening ``eps_c`` -- the live run's
internal kernel) in the static potential prepare.py fitted to the live model
(NFW + MN3 disc + Hernquist bulge, gcmw.static_config_params). No stripping
bookkeeping: a stripped star is just a star.

The span is cut into --chunks calls; each chunk's end state is saved (chunk_*.npz), so a
rerun continues from the last one. Writes rows.jsonl (time, energy, angular momentum).

usage: python run_static.py --prepare runs/prepare --M 4096 --out runs/pal5_static [--small]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcmw  # noqa: E402



def _mass_arg(v):
    """'present' / 'initial' or a number of Msun."""
    return v if v in ("present", "initial") else float(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", required=True)
    p.add_argument("--catalogue", default=os.environ.get("ODISSEO_GC_DATA"))
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--model", default="king", choices=("king", "plummer"))
    p.add_argument("--mass", default="present",
                   help="'present' (catalogue), 'initial' (its lg M_ini) or Msun (globular_clusters.cluster_table)")
    p.add_argument("--r-half-pc", type=float, default=None, help="3D half-mass radius [pc] instead of the catalogue's")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-gyr", type=float, default=None)
    p.add_argument("--dt", type=float, default=2.5e-4, help="leapfrog step (code units)")
    p.add_argument("--chunks", type=int, default=None,
                   help="default: one chunk per live segment (the static run's chunk ends are the live edges)")
    p.add_argument("--live-dt", type=float, default=1e-3, help="the live run's dt (its span and edges)")
    p.add_argument("--live-steps", type=int, default=100, help="the live run's steps per segment")
    p.add_argument("--eps-c", type=float, default=None)
    p.add_argument("--small", action="store_true", help="CPU small mode: M 64, 2 Myr, 2 chunks")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.small:
        a.M, a.t_gyr, a.chunks = min(a.M, 64), a.t_gyr or 0.002, 2
        a.live_steps = 5

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from odisseo import globular_clusters as gc
    from odisseo.dynamics import DIRECT_ACC_MATRIX
    from odisseo.time_integration import time_integration
    from odisseo.utils import E_tot, Angular_momentum

    with open(os.path.join(a.prepare, "prepare.json")) as fh:
        prep = json.load(fh)
    os.makedirs(a.out, exist_ok=True)
    cat = gc.load_baumgardt(directory=a.catalogue)
    centres = (np.array(prep["back"]["x"]), np.array(prep["back"]["v"]))
    xc, vc, mc, table = gc.clusters_from_catalogue(prep["names"], a.M, model=a.model, catalogue=cat,
                                                   centres=centres, seed=a.seed, mass=_mass_arg(a.mass), r_half_pc=a.r_half_pc, return_table=True)
    eps_c = a.eps_c or gc.default_cluster_softening(table)
    t_gyr = a.t_gyr if a.t_gyr is not None else prep["t_gyr"]
    T = t_gyr * 1e3 / gcmw.TIME_MYR
    # the live run's span exactly (run_live.py: whole segments of --live-steps at --live-dt)
    seg_live = a.live_steps * a.live_dt
    n_seg = max(1, int(round(T / seg_live)))
    T = n_seg * seg_live
    if a.chunks is None:
        a.chunks = n_seg
    t_chunk = T / a.chunks
    steps_chunk = max(1, int(round(t_chunk / a.dt)))
    a.dt = t_chunk / steps_chunk
    n = xc.shape[0] * xc.shape[1]
    config, params = gcmw.static_config_params(
        prep["fit_params"], N_particles=n, num_timesteps=steps_chunk, softening=eps_c, softening_kernel="plummer",
        acceleration_scheme=DIRECT_ACC_MATRIX, return_snapshots=False,
    )
    params = params._replace(t_end=t_chunk)
    gcmw.write_json(os.path.join(a.out, "run.json"), dict(
        names=prep["names"], M=a.M, model=a.model, seed=a.seed, t_gyr=t_gyr, dt=a.dt, chunks=a.chunks,
        steps_per_chunk=steps_chunk, eps_c=eps_c, static_params=prep["fit_params"], backend=jax.default_backend(),
        mass=a.mass, r_half_pc=a.r_half_pc,
        table=[dict(name=r["name"], mass=r["mass"], mass_msun=r["mass_msun"], r_half_pc=r["r_half_pc"], w0=r["w0"])
               for r in table],
        device=str(jax.devices()[0]), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")))
    mass = jnp.asarray(mc.reshape(-1))
    done = sorted(glob.glob(os.path.join(a.out, "chunk_*.npz")))
    if done:
        d = np.load(done[-1])
        state, k0 = jnp.asarray(d["state"]), int(d["chunk"]) + 1
        print(f"[static] continuing after chunk {k0 - 1}", flush=True)
    else:
        state, k0 = jnp.stack([jnp.asarray(xc.reshape(-1, 3)), jnp.asarray(vc.reshape(-1, 3))], 1), 0
        np.savez(os.path.join(a.out, "chunk_-0001.npz"), state=np.asarray(state), chunk=-1, time=0.0)

    def diag(s, t):
        e = float(jnp.sum(E_tot(s, mass, config, params)))
        lz = np.asarray(jnp.sum(Angular_momentum(s, mass), axis=0)).tolist()
        return dict(time=t, energy=e, angular_momentum=lz)

    rows = os.path.join(a.out, "rows.jsonl")
    if k0 == 0:
        with open(rows, "w") as fh:
            fh.write(json.dumps(dict(chunk=-1, **diag(state, 0.0))) + "\n")
    for k in range(k0, a.chunks):
        t0 = time.time()
        state = time_integration(state, mass, config, params)
        state.block_until_ready()
        t = (k + 1) * t_chunk
        np.savez(os.path.join(a.out, f"chunk_{k:04d}.npz"), state=np.asarray(state), chunk=k, time=t)
        row = dict(chunk=k, wall_s=time.time() - t0, **diag(state, t))
        with open(rows, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[static] chunk {k} t {t:.4f} E {row['energy']:.6e} ({row['wall_s']:.1f} s)", flush=True)
    gcmw.write_json(os.path.join(a.out, "done.json"), dict(time=a.chunks * t_chunk, chunks=a.chunks))


if __name__ == "__main__":
    main()
