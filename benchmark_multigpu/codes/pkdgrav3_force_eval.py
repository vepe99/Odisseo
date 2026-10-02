"""Force-evaluation benchmark script, run INSIDE pkdgrav3's embedded interpreter.

Invoke as ``pkdgrav3 [-sz CORES] pkdgrav3_force_eval.py``; importing ``PKDGRAV``
puts the binary in analysis mode instead of simulation mode, which is what lets
us time exactly one force evaluation on exactly the particles jaccpot sees --
no integrator, no I/O, no timestep policy in the measured region.

Configuration comes from the environment (pkdgrav3 owns argv), see ``_cfg``.
Results are written to ``$PKD_OUT`` as an ``.npz`` per theta plus a JSON summary.

Timed regions, both reported, because the two codes draw the line differently:
  * ``gravity``   -- the tree walk + force evaluation alone.
  * ``force_eval`` -- domain_decompose + build_tree + gravity, i.e. everything
    jaccpot's ``distributed_fmm_accelerations`` does per call (it rebuilds the
    tree every call, so this is the honest like-for-like number).
"""

from __future__ import annotations

import json
import os
import resource
import time

import numpy as np

import PKDGRAV as pkd

# The field selectors are members of the PKD_FIELD enum, not module-level names
# (they come from the Cython `cdef extern` enum in modules/PKDGRAV.pxd).
FIELD_ACCELERATION = pkd.PKD_FIELD.FIELD_ACCELERATION
FIELD_MASS = pkd.PKD_FIELD.FIELD_MASS
FIELD_POSITION = pkd.PKD_FIELD.FIELD_POSITION
FIELD_POTENTIAL = pkd.PKD_FIELD.FIELD_POTENTIAL


def _cfg():
    e = os.environ
    return dict(
        ic=e["PKD_IC"],
        out=e["PKD_OUT"],
        thetas=[float(x) for x in e.get("PKD_THETAS", "0.7").split(",")],
        repeats=int(e.get("PKD_REPEATS", "5")),
        warmup=int(e.get("PKD_WARMUP", "1")),
        soft=float(e.get("PKD_SOFT", "0.0")),
        n_bucket=int(e.get("PKD_NBUCKET", "16")),
        n_group=int(e.get("PKD_NGROUP", "0")),  # 0 -> leave pkdgrav3's default
        gpu=e.get("PKD_GPU", "1") not in ("0", "false", "False"),
        save_arrays=e.get("PKD_SAVE_ARRAYS", "1") not in ("0", "false", "False"),
    )


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    cfg = _cfg()

    params = dict(
        # isolated, non-cosmological: pkdgrav3 defaults bPeriodic/bComove to False
        # already, but be explicit -- these decide whether Ewald is in the walk.
        bPeriodic=False,
        bComove=False,
        bEwald=False,
        bDoGravity=True,
        # memory model: without these the fields simply do not exist and
        # get_array() returns zeros.
        bMemAcceleration=True,
        bMemPotential=True,
        bMemMass=True,
        bMemSoft=True,
        nBucket=cfg["n_bucket"],
        dSoft=cfg["soft"],
        bGPU=cfg["gpu"],
        # keep particle ids so reorder() can restore the input ordering
        bMemUnordered=False,
        bMemParticleID=True,
    )
    if cfg["n_group"] > 0:
        params["nGroup"] = cfg["n_group"]

    t_load0 = time.perf_counter()
    sim_time = pkd.load(cfg["ic"], **params)
    t_load = time.perf_counter() - t_load0

    n_particles = None
    results = []
    for theta in cfg["thetas"]:
        samples = []
        for i in range(cfg["warmup"] + cfg["repeats"]):
            t0 = time.perf_counter()
            pkd.domain_decompose()
            t1 = time.perf_counter()
            pkd.build_tree(ewald=False)
            t2 = time.perf_counter()
            pkd.gravity(
                time=sim_time,
                delta=0.0,
                theta=theta,
                ewald=False,
                kick_close=False,
                kick_open=False,
            )
            t3 = time.perf_counter()
            if i >= cfg["warmup"]:
                samples.append(
                    dict(
                        domain_s=t1 - t0,
                        tree_s=t2 - t1,
                        gravity_s=t3 - t2,
                        force_eval_s=t3 - t0,
                    )
                )

        def stat(key):
            v = np.array([s[key] for s in samples], float)
            return dict(min=float(v.min()), mean=float(v.mean()), std=float(v.std()))

        entry = dict(
            theta=theta,
            repeats=cfg["repeats"],
            warmup=cfg["warmup"],
            domain_s=stat("domain_s"),
            tree_s=stat("tree_s"),
            gravity_s=stat("gravity_s"),
            force_eval_s=stat("force_eval_s"),
        )

        if cfg["save_arrays"]:
            # reorder() restores the input file ordering, so row i here is row i
            # of the Tipsy file and therefore row i of the jaccpot arrays.
            pkd.reorder()
            acc = pkd.get_array(FIELD_ACCELERATION)
            pot = pkd.get_array(FIELD_POTENTIAL)
            pos = pkd.get_array(FIELD_POSITION)
            mass = pkd.get_array(FIELD_MASS)
            n_particles = int(mass.shape[0])
            path = f"{cfg['out']}_theta{theta:g}.npz"
            np.savez_compressed(
                path, acc=acc, pot=pot, pos=pos, mass=mass, theta=np.float64(theta)
            )
            entry["arrays"] = path

        results.append(entry)
        print(
            f"[force_eval] theta={theta:g}  force_eval={entry['force_eval_s']['min']*1e3:.2f} ms"
            f"  (domain {entry['domain_s']['min']*1e3:.2f} + tree {entry['tree_s']['min']*1e3:.2f}"
            f" + gravity {entry['gravity_s']['min']*1e3:.2f})",
            flush=True,
        )

    summary = dict(
        ic=cfg["ic"],
        n_particles=n_particles,
        sim_time=float(sim_time),
        load_s=t_load,
        softening=cfg["soft"],
        n_bucket=cfg["n_bucket"],
        n_group=cfg["n_group"],
        gpu=cfg["gpu"],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        peak_rss_mib=_peak_rss_mib(),
        results=results,
    )
    with open(f"{cfg['out']}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[force_eval] wrote {cfg['out']}_summary.json", flush=True)


main()
