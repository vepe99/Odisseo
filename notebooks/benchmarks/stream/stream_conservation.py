"""Energy / momentum evolution for the GD1 stream benchmark (diagnostic, not timed).

Re-runs the same problem as stream_benchmark.py (same ICs, potentials, dt, 1000
leapfrog steps) for each code, saving phase-space snapshots every SNAP_EVERY
steps. Conserved quantities are then computed for every code with ONE common
float64 estimator (numpy direct-sum softened self-potential + Odisseo's external
potential), so differences are due to the integrators only.

    PY=../../../.venv/bin/python ; CPY=.venv_cpu/bin/python
    $PY  stream_conservation.py run --code odisseo     --N 500 2000
    $PY  stream_conservation.py run --code odisseo_cpu --N 500 2000 --core 63
    $CPY stream_conservation.py run --code gala        --N 500 2000 --core 62
    $CPY stream_conservation.py run --code galpy       --N 500 2000 --core 61
    $PY  stream_conservation.py analyze

Outputs in results/conservation/: traj_{code}_N{N}.npz (t, pos, vel snapshots),
conservation_{code}_N{N}.npz (E, KE, W, Phi_ext, P, L vs t), conservation.{pdf,png}.

Note: the external potential (NFW + MN + PSP) is static and axisymmetric about z,
so E and L_z are conserved; P and L_x, L_y are not (the host exerts a torque/force).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import stream_benchmark as sb

OUT = sb.RESULTS / "conservation"
SNAP_EVERY = 10


def traj_path(code, N):
    return OUT / f"traj_{code}_N{N}.npz"


def save_traj(code, N, xs, vs, ic, wall, extra=""):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ic["meta"]
    steps = np.arange(len(xs)) * SNAP_EVERY
    np.savez(traj_path(code, N), t=steps * meta["dt"], step=steps, pos=np.asarray(xs), vel=np.asarray(vs),
             mass=ic["mass"], wall_s=wall, meta=json.dumps(meta), note=extra)
    # consistency with the timed run's final state
    f = sb.RESULTS / f"final_{code}_N{N}.npz"
    if f.exists():
        with np.load(f) as d:
            dx = np.median(np.linalg.norm(np.asarray(xs[-1]) - d["pos"], axis=1))
        print(f"[{code}] N={N} median |x_final - benchmark final| = {dx:.2e} (code units)", flush=True)
    print(f"[{code}] N={N} saved {len(xs)} snapshots in {wall:.1f}s -> {traj_path(code, N)}", flush=True)


# --------------------------------------------------------------------------- #
def run_odisseo(a, device, fp32=False):
    code = "odisseo_fp32" if fp32 else ("odisseo" if device == "gpu" else "odisseo_cpu")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()
        if a.core is not None:
            os.sched_setaffinity(0, {a.core})
    else:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        from autocvd import autocvd
        autocvd(num_gpus=1, timeout=a.gpu_timeout)
    import jax
    if not fp32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    dtype = np.float32 if fp32 else np.float64
    from odisseo import construct_initial_state
    from odisseo.time_integration import time_integration
    dev = jax.devices()[0]
    assert dev.platform == device, (dev.platform, device)
    print(f"[{code}] device {dev}", flush=True)
    for N in a.N:
        ic = sb.load_ic(N); meta = ic["meta"]
        _, config, params = sb.odisseo_setup(N, x64=not fp32)
        # Odisseo's fixed-step leapfrog is stateless between steps (fresh force eval each
        # KDK step), so integrating 100 chunks of 10 steps == one 1000-step run.
        config = config._replace(num_timesteps=SNAP_EVERY)
        params = params._replace(t_end=meta["t_end"] * SNAP_EVERY / meta["n_steps"])
        state = jax.device_put(construct_initial_state(jnp.asarray(ic["pos"], dtype), jnp.asarray(ic["vel"], dtype)), dev)
        mass = jax.device_put(jnp.asarray(ic["mass"], dtype), dev)
        step = time_integration.lower(state, mass, config, params).compile()
        xs, vs = [ic["pos"]], [ic["vel"]]
        t0 = time.perf_counter()
        for _ in range(meta["n_steps"] // SNAP_EVERY):
            state = step(state, mass, params)
            s = np.asarray(state); assert s.dtype == dtype, s.dtype
            s = s.astype(np.float64); xs.append(s[:, 0]); vs.append(s[:, 1])
        save_traj(code, N, xs, vs, ic, time.perf_counter() - t0)


def run_gala(a):
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.dynamics.nbody import DirectNBody
    from gala.integrate import LeapfrogIntegrator
    sb.check_cpu_potentials()
    us = sb.gala_units()
    for N in a.N:
        ic = sb.load_ic(N); meta = ic["meta"]
        w0 = gd.PhaseSpacePosition(pos=ic["pos"].T * us["length"], vel=ic["vel"].T * us["length"] / us["time"])
        pps = [gp.PlummerPotential(m=m, b=meta["softening"], units=us) for m in ic["mass"]]
        sim = DirectNBody(w0, pps, external_potential=sb.gala_potential(meta), units=us, save_all=True)
        t0 = time.perf_counter()
        w = sim.integrate_orbit(Integrator=LeapfrogIntegrator, dt=meta["dt"], t1=0.0, n_steps=meta["n_steps"])
        wall = time.perf_counter() - t0
        x = np.asarray(w.xyz.decompose(us).value)        # (3, n_t, N)
        v = np.asarray(w.v_xyz.decompose(us).value)
        x = np.moveaxis(x, 0, -1)[::SNAP_EVERY]; v = np.moveaxis(v, 0, -1)[::SNAP_EVERY]
        save_traj("gala", N, list(x), list(v), ic, wall)


def run_galpy(a):
    """Same integrator as the timed benchmark: numpy direct sum + galpy external forces, KDK."""
    sb.check_cpu_potentials()
    for N in a.N:
        ic = sb.load_ic(N); meta = ic["meta"]
        ext = sb.GalpyExternal(meta); mass = ic["mass"]
        G, eps, dt = meta["G"], meta["softening"], meta["dt"]
        x, v = ic["pos"].copy(), ic["vel"].copy()
        xs, vs = [x.copy()], [v.copy()]
        t0 = time.perf_counter()
        acc = sb.direct_acc_numpy(x, mass, eps, G) + ext.acc(x)
        for i in range(meta["n_steps"]):
            v += 0.5 * dt * acc
            x += dt * v
            acc = sb.direct_acc_numpy(x, mass, eps, G) + ext.acc(x)
            v += 0.5 * dt * acc
            if (i + 1) % SNAP_EVERY == 0:
                xs.append(x.copy()); vs.append(v.copy())
        save_traj("galpy", N, xs, vs, ic, time.perf_counter() - t0)


def cmd_run(a):
    if a.code in ("gala", "galpy"):
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"
        if a.core is not None:
            os.sched_setaffinity(0, {a.core})
        (run_gala if a.code == "gala" else run_galpy)(a)
    else:
        run_odisseo(a, "cpu" if a.code == "odisseo_cpu" else "gpu", fp32=a.code == "odisseo_fp32")


# --------------------------------------------------------------------------- #
def diagnostics(code, N):
    out = OUT / f"conservation_{code}_N{N}.npz"
    src = traj_path(code, N)
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return dict(np.load(out))
    d = np.load(src); meta = json.loads(str(d["meta"])); m = d["mass"]
    KE, W, PHI, P, L = [], [], [], [], []
    for x, v in zip(d["pos"], d["vel"]):
        _, phi = sb.odisseo_external(x)
        KE.append(0.5 * np.sum(m * np.sum(v * v, 1)))
        W.append(sb.self_potential_energy(x, m, meta["softening"], meta["G"]))
        PHI.append(float(np.sum(m * phi)))
        P.append(np.sum(m[:, None] * v, 0))
        L.append(np.sum(m[:, None] * np.cross(x, v), 0))
    KE, W, PHI = map(np.array, (KE, W, PHI))
    res = dict(t=d["t"], t_Gyr=d["t"] * meta["phys"]["code_time_Gyr"], KE=KE, W=W, PHI=PHI,
               E=KE + W + PHI, P=np.array(P), L=np.array(L))
    np.savez(out, **res)
    return res


def cmd_analyze(a):
    os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = a.N
    cols = [("dE", r"$|E(t)-E_0|\,/\,|E_0|$", True),
            ("dLz", r"$|L_z(t)-L_{z,0}|\,/\,|L_0|$", True)]
    # Odisseo (GPU, CPU) and galpy agree to <= 3e-8 in E: draw galpy as a wide translucent band
    # underneath and Odisseo on top, otherwise the last-drawn curve hides the others.
    STYLE = {"galpy": dict(lw=9.0, alpha=0.45, zorder=1), "gala": dict(lw=2.5, zorder=2),
             "odisseo": dict(lw=2.5, zorder=4), "odisseo_fp32": dict(lw=2.5, ls="-.", zorder=3), "odisseo_cpu": dict(lw=2.5, ls=(0, (4, 3)), zorder=5)}
    FS = 30   # label / tick size, as in the docs scaling plot (notebooks/scalability)
    fig, axes = plt.subplots(len(cols), len(Ns), figsize=(9 * len(Ns), 6.5 * len(cols)),
                             sharex=True, squeeze=False)
    rows = []
    for j, N in enumerate(Ns):
        for code in sb.CODES:
            if not traj_path(code, N).exists():
                print(f"[analyze] missing {traj_path(code, N)}"); continue
            r = diagnostics(code, N)
            L0 = np.linalg.norm(r["L"][0])
            series = dict(dE=np.abs(r["E"] - r["E"][0]) / abs(r["E"][0]),
                          dLz=np.abs(r["L"][:, 2] - r["L"][0, 2]) / L0,
                          P=np.linalg.norm(r["P"], axis=1) / np.linalg.norm(r["P"][0]),
                          L=np.linalg.norm(r["L"], axis=1) / L0)
            for i, (k, lab, logy) in enumerate(cols):
                ax = axes[i, j]
                y = series[k]
                if logy:
                    y = np.where(y > 0, y, np.nan)
                ax.plot(r["t_Gyr"], y, color=sb.COLOR[code], label=sb.LABEL[code],
                        **{"ls": "-", **STYLE[code]})
                if logy:
                    ax.set_yscale("log")
                if j == 0:
                    ax.set_ylabel(lab, fontsize=FS)
            rows.append(dict(code=code, N=N, dE_final=series["dE"][-1], dE_max=np.nanmax(series["dE"]),
                             dLz_final=series["dLz"][-1], dLz_max=np.nanmax(series["dLz"])))
        axes[0, j].set_title(f"N = {N}", fontsize=FS)
        axes[-1, j].set_xlabel("t [Gyr]", fontsize=FS)
    for ax in axes.flat:
        ax.grid(alpha=0.3, lw=0.5); ax.tick_params(axis="both", which="major", labelsize=FS)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=25, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"conservation.{ext}", bbox_inches="tight", dpi=300)
    import csv
    with open(OUT / "conservation_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['code']:12s} N={r['N']:5d} |dE/E| final {r['dE_final']:.2e} max {r['dE_max']:.2e}   "
              f"|dLz|/|L| final {r['dLz_final']:.2e} max {r['dLz_max']:.2e}")
    print(f"[analyze] wrote {OUT / 'conservation.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--code", choices=sb.CODES, required=True)
    r.add_argument("--N", type=int, nargs="+", default=[500, 2000]); r.add_argument("--core", type=int)
    r.add_argument("--gpu_timeout", type=float, default=3600)
    an = sp.add_parser("analyze"); an.add_argument("--N", type=int, nargs="+", default=[500, 2000])
    a = p.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
