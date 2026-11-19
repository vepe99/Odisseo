"""GD1 stream, direct N-body: Odisseo (GPU and 1 CPU core) vs gala and galpy (1 CPU core).

One file, four subcommands. gala and galpy are not in the Odisseo venv, so the
``run --code gala|galpy`` step uses a separate venv (see README.md); every other
step uses the main Odisseo venv.

    PY=../../../.venv/bin/python          # Odisseo venv
    CPY=.venv_cpu/bin/python              # gala + galpy venv

    $PY  stream_benchmark.py setup   --N 500 1000 2000 4000 8000 16000 32000 64000
    $PY  stream_benchmark.py run --code odisseo     --N 500 1000 2000 4000 8000 16000 32000 64000
    $PY  stream_benchmark.py run --code odisseo_cpu --N 500 --core 63      # one process per N/core
    $CPY stream_benchmark.py run --code gala        --N 500 1000 2000 4000 8000
    $CPY stream_benchmark.py run --code galpy       --N 500 1000 2000 4000 8000
    $PY  stream_benchmark.py analyze

Problem (notebooks/GD1.ipynb): Plummer cluster (M = 10^4.05 Msun, a = 100 pc)
placed on the GD1 progenitor orbit integrated back 3 Gyr, evolved forward 3 Gyr
with direct-summation self-gravity (Plummer softening 0.1 pc) in
NFW + Miyamoto-Nagai + power-law-cutoff bulge (MWPotential2014). Every code:
fp64, fixed-step second-order leapfrog, 1000 steps (dt = 3 Myr), same IC files.

Outputs in results/: stream_timings.csv (one row per run), final_{code}_N{N}.npz,
stream_summary.csv, stream_timing.{pdf,png}.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

# --------------------------------------------------------------------------- #
# Physical setup (identical to notebooks/GD1.ipynb)                           #
# --------------------------------------------------------------------------- #
PHYS = dict(
    code_length_kpc=10.0, code_mass_Msun=1e4, code_time_Gyr=3.0, t_end_Gyr=3.0,
    softening_pc=0.1, plummer_logM=4.05, plummer_a_pc=100.0,
    MN_M_Msun=68_193_902_782.346756, MN_a_kpc=3.0, MN_b_kpc=0.280,
    NFW_M_Msun=4.3683325e11, NFW_rs_kpc=16.0,
    PSP_M_Msun=4501365375.06545, PSP_alpha=1.8, PSP_rc_kpc=1.9,
    com_pos_kpc=[11.8, 0.79, 6.4], com_vel_kms=[109.5, -254.5, -90.3],
    plummer_key=500000000,
)
N_STEPS = 1000          # dt = 3 Myr (the notebook's num_timesteps)
N_STEPS_COM = 100_000   # backward progenitor orbit (shared IC, cost irrelevant)

CSV_COLS = ["timestamp_utc", "code", "device", "N", "precision", "scheme", "repeat", "status",
            "n_steps", "dt", "force_evals_per_step", "n_force_evals", "wall_s", "setup_s",
            "interactions_per_s", "threads", "loadavg_before", "loadavg_after", "host", "cpu_model",
            "versions", "error"]

CODES = ["odisseo", "odisseo_fp32", "odisseo_cpu", "gala", "galpy"]
LABEL = {"odisseo": "Odisseo (1×A100 GPU)", "odisseo_fp32": "Odisseo fp32 (1×A100 GPU)", "odisseo_cpu": "Odisseo (1 CPU core)",
         "gala": "gala (1 CPU core)", "galpy": "galpy (1 CPU core)"}
COLOR = {"odisseo": "#2a6fdb", "odisseo_fp32": "#1b9e77", "odisseo_cpu": "#7aa7ee", "gala": "#d1495b", "galpy": "#edae49"}
MARKER = {"odisseo": "o", "odisseo_fp32": "P", "odisseo_cpu": "s", "gala": "D", "galpy": "^"}
LS = {"odisseo": "-", "odisseo_fp32": "-.", "odisseo_cpu": "--", "gala": "-", "galpy": "-"}


# --------------------------------------------------------------------------- #
# Small helpers (numpy only)                                                   #
# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def ic_path(N: int) -> Path:
    return DATA / f"stream_N{N}.npz"


def load_ic(N: int) -> dict:
    with np.load(ic_path(N)) as f:
        d = {k: f[k] for k in ("pos", "vel", "mass")}
        meta = json.loads(str(f["meta"]))
    if sha256_arrays(d["pos"], d["vel"], d["mass"]) != meta["sha256"]:
        raise RuntimeError(f"checksum mismatch for {ic_path(N)}")
    d["meta"] = meta
    return d


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})
        f.flush(); os.fsync(f.fileno())


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def self_potential_energy(pos, mass, eps, G, threads=8) -> float:
    """W = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2), float64, blocked + threaded."""
    from concurrent.futures import ThreadPoolExecutor
    n = len(mass)
    x, y, z = (np.ascontiguousarray(pos[:, k], np.float64) for k in range(3))
    block = max(1, (1 << 21) // n)

    def work(i0):
        i1 = min(n, i0 + block)
        r2 = (x[i0:i1, None] - x) ** 2 + (y[i0:i1, None] - y) ** 2 + (z[i0:i1, None] - z) ** 2
        zero = r2 == 0.0
        inv = 1.0 / np.sqrt(r2 + eps * eps)
        inv[zero] = 0.0
        return float(mass[i0:i1] @ (inv @ mass))

    with ThreadPoolExecutor(threads) as ex:
        return -0.5 * G * sum(ex.map(work, range(0, n, block)))


# --------------------------------------------------------------------------- #
# Odisseo                                                                      #
# --------------------------------------------------------------------------- #
def odisseo_setup(N: int, x64: bool = True):
    """(code_units, config, params) for Odisseo, fp64 unless x64=False. Imports JAX."""
    import jax
    if x64:
        jax.config.update("jax_enable_x64", True)
    from astropy import units as u
    import odisseo.option_classes as oc
    from odisseo.units import CodeUnits
    p = PHYS
    cu = CodeUnits(p["code_length_kpc"] * u.kpc, p["code_mass_Msun"] * u.Msun, G=1,
                   unit_time=p["code_time_Gyr"] * u.Gyr)
    L, M, T = cu.code_length, cu.code_mass, cu.code_time
    config = oc.SimulationConfig(
        N_particles=N, return_snapshots=False, num_timesteps=N_STEPS,
        external_accelerations=(oc.NFW_POTENTIAL, oc.MN_POTENTIAL, oc.PSP_POTENTIAL),
        acceleration_scheme=oc.DIRECT_ACC, integrator=oc.LEAPFROG, fixed_timestep=True,
        softening=(p["softening_pc"] * u.pc).to(L).value, double_map=False)
    params = oc.SimulationParams(
        t_end=(p["t_end_Gyr"] * u.Gyr).to(T).value,
        Plummer_params=oc.PlummerParams(Mtot=(10 ** p["plummer_logM"] * u.Msun).to(M).value,
                                        a=(p["plummer_a_pc"] * u.pc).to(L).value),
        MN_params=oc.MNParams(M=(p["MN_M_Msun"] * u.Msun).to(M).value,
                              a=(p["MN_a_kpc"] * u.kpc).to(L).value,
                              b=(p["MN_b_kpc"] * u.kpc).to(L).value),
        NFW_params=oc.NFWParams(Mvir=(p["NFW_M_Msun"] * u.Msun).to(M).value,
                                r_s=(p["NFW_rs_kpc"] * u.kpc).to(L).value),
        PSP_params=oc.PSPParams(M=(p["PSP_M_Msun"] * u.Msun).to(M).value, alpha=p["PSP_alpha"],
                                r_c=(p["PSP_rc_kpc"] * u.kpc).to(L).value),
        G=cu.G)
    return cu, config, params


def odisseo_external(pos):
    """Odisseo's external acceleration and potential at pos (code units), on the CPU."""
    from odisseo.potentials import NFW, MyamotoNagai, PowerSphericalPotentialwCutoff
    import jax.numpy as jnp
    _, config, params = odisseo_setup(1)
    state = jnp.asarray(np.stack([pos, np.zeros_like(pos)], axis=1))
    acc = np.zeros_like(pos); pot = np.zeros(len(pos))
    for f in (NFW, MyamotoNagai, PowerSphericalPotentialwCutoff):
        a_, p_ = f(state, config, params, return_potential=True)
        acc += np.asarray(a_); pot += np.asarray(p_)
    return acc, pot


def cmd_setup(a) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from astropy import units as u
    from odisseo import construct_initial_state
    from odisseo.initial_condition import Plummer_sphere
    from odisseo.time_integration import time_integration

    DATA.mkdir(parents=True, exist_ok=True)
    cu, config, params = odisseo_setup(1)
    cfg_com = config._replace(N_particles=1, num_timesteps=N_STEPS_COM)
    prm_com = params._replace(t_end=-params.t_end)
    pos_f = jnp.array([PHYS["com_pos_kpc"]]) * u.kpc.to(cu.code_length)
    vel_f = jnp.array([PHYS["com_vel_kms"]]) * (u.km / u.s).to(cu.code_velocity)
    com = np.asarray(time_integration(construct_initial_state(pos_f, vel_f),
                                      jnp.array([params.Plummer_params.Mtot]), cfg_com, prm_com))
    common = dict(G=float(cu.G), softening=float(config.softening), t_end=float(params.t_end),
                  n_steps=N_STEPS, dt=float(params.t_end / N_STEPS),
                  code_velocity_kms=float((1 * cu.code_velocity).to(u.km / u.s).value),
                  plummer_M=float(params.Plummer_params.Mtot), plummer_a=float(params.Plummer_params.a),
                  MN=[float(params.MN_params.M), float(params.MN_params.a), float(params.MN_params.b)],
                  NFW=[float(params.NFW_params.Mvir), float(params.NFW_params.r_s)],
                  PSP=[float(params.PSP_params.M), float(params.PSP_params.alpha), float(params.PSP_params.r_c)],
                  com_pos=com[0, 0].tolist(), com_vel=com[0, 1].tolist(), phys=PHYS,
                  jax_version=jax.__version__)
    for N in a.N:
        if ic_path(N).exists():
            print(f"[setup] exists: {ic_path(N)}"); continue
        pos, vel, mass = Plummer_sphere(jax.random.PRNGKey(PHYS["plummer_key"]),
                                        config._replace(N_particles=N), params)
        pos = np.asarray(pos, np.float64) + com[0, 0]
        vel = np.asarray(vel, np.float64) + com[0, 1]
        mass = np.asarray(mass, np.float64)
        meta = dict(common, N=N, sha256=sha256_arrays(pos, vel, mass))
        np.savez(ic_path(N), pos=pos, vel=vel, mass=mass, meta=json.dumps(meta))
        print(f"[setup] wrote {ic_path(N)}")

    # Odisseo's external field at random points (0.3-30 kpc): gala/galpy check against it.
    rng = np.random.default_rng(0)
    r = 10 ** rng.uniform(-1.5, 0.5, 2000)
    v = rng.normal(size=(2000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    pts = r[:, None] * v
    acc, pot = odisseo_external(pts)
    np.savez(DATA / "potential_check.npz", pos=pts, acc=acc, pot=pot, meta=json.dumps(common))
    print(f"[setup] wrote {DATA / 'potential_check.npz'}")


def run_odisseo(a, device: str, fp32: bool = False) -> None:
    code = "odisseo_fp32" if fp32 else ("odisseo" if device == "gpu" else "odisseo_cpu")
    prec = "fp32" if fp32 else "fp64"
    if device == "cpu":
        # one thread, pinned to one core: same footing as gala / galpy
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1] if a.core is None else a.core})
        devname = f"cpu core {sorted(os.sched_getaffinity(0))}"
    else:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        from autocvd import autocvd
        autocvd(num_gpus=1, timeout=a.gpu_timeout)
        import subprocess
        idx = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
        devname = subprocess.run(["nvidia-smi", f"--id={idx}", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

    import jax
    import jax.numpy as jnp
    if not fp32:
        jax.config.update("jax_enable_x64", True)
    from odisseo import construct_initial_state
    from odisseo.time_integration import time_integration
    import odisseo.option_classes as oc
    dtype = np.float32 if fp32 else np.float64

    dev = jax.devices()[0]
    assert dev.platform == device, (dev.platform, device)
    print(f"[{code}] device {devname} ({dev})", flush=True)
    versions = json.dumps(dict(jax=jax.__version__, numpy=np.__version__,
                               python=platform.python_version(), device=devname))

    for N in a.N:
        ic = load_ic(N); meta = ic["meta"]
        _, config, params = odisseo_setup(N, x64=not fp32)
        n_steps = a.n_steps or meta["n_steps"]
        if a.n_steps:
            config = config._replace(num_timesteps=n_steps)
            params = params._replace(t_end=n_steps * meta["dt"])
        state = jax.device_put(construct_initial_state(jnp.asarray(ic["pos"], dtype), jnp.asarray(ic["vel"], dtype)), dev)
        mass = jax.device_put(jnp.asarray(ic["mass"], dtype), dev)
        if device == "cpu":
            reps = 3 if N <= 1000 else 1
        else:
            reps = 3 if N >= 32000 else 5
        # DIRECT_ACC first; DIRECT_ACC_LAXMAP only if it fails (e.g. OOM at N = 64000 on 40 GB)
        for sch in ("DIRECT_ACC", "DIRECT_ACC_LAXMAP"):
            cfg = config._replace(acceleration_scheme=getattr(oc, sch), batch_size=a.batch_size)
            base = dict(code=code, device=device, N=N, precision=prec, scheme=sch, n_steps=n_steps,
                        dt=meta["dt"], force_evals_per_step=2, n_force_evals=2 * n_steps, threads=1,
                        host=platform.node(), cpu_model=cpu_model(), versions=versions)
            try:
                t0 = time.perf_counter()
                compiled = time_integration.lower(state, mass, cfg, params).compile()
                t_comp = time.perf_counter() - t0
                if device == "gpu":   # warm-up
                    compiled(state, mass, params).block_until_ready()
                for r in range(reps):
                    la0 = os.getloadavg()[0]
                    t0 = time.perf_counter()
                    out = compiled(state, mass, params)
                    out.block_until_ready()
                    wall = time.perf_counter() - t0
                    ips = N * N * 2 * n_steps / wall
                    if not a.n_steps:
                        append_row(a.csv, dict(base, timestamp_utc=utcnow(), repeat=r, status="ok",
                                               wall_s=wall, setup_s=t_comp, interactions_per_s=ips,
                                               loadavg_before=la0, loadavg_after=os.getloadavg()[0]))
                    print(f"[{code}] {sch:18s} N={N:6d} rep={r} wall={wall:9.3f}s "
                          f"compile={t_comp:5.1f}s int/s={ips:.3e}", flush=True)
                o = np.asarray(out)
                assert o.dtype == dtype, o.dtype
                o = o.astype(np.float64)
                _save_final(a, code, N, o[:, 0], o[:, 1], ic, wall)
                break
            except Exception as e:
                status = "oom" if "RESOURCE_EXHAUSTED" in str(e) else "error"
                if not a.n_steps:
                    append_row(a.csv, dict(base, timestamp_utc=utcnow(), status=status, error=repr(e)[:500]))
                print(f"[{code}] {sch} N={N} {status}: {str(e)[:200]}", flush=True)
                jax.clear_caches()


def _save_final(a, code, N, pos, vel, ic, wall):
    out = (RESULTS if not a.n_steps else Path(a.out_dir)) / f"final_{code}_N{N}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pos=pos, vel=vel, mass=ic["mass"], wall_s=wall, meta=json.dumps(ic["meta"]))


# --------------------------------------------------------------------------- #
# gala and galpy (separate venv)                                               #
# --------------------------------------------------------------------------- #
def gala_units():
    from astropy import units as u
    from gala.units import UnitSystem
    return UnitSystem(PHYS["code_length_kpc"] * u.kpc, PHYS["code_mass_Msun"] * u.Msun,
                      PHYS["code_time_Gyr"] * u.Gyr, u.radian)


def gala_potential(meta):
    """NFW + Miyamoto-Nagai + PowerLawCutoff, same parameters as Odisseo (code units)."""
    import gala.potential as gp
    us = gala_units()
    pot = gp.CCompositePotential()
    pot["nfw"] = gp.NFWPotential(m=meta["NFW"][0], r_s=meta["NFW"][1], units=us)
    pot["disk"] = gp.MiyamotoNagaiPotential(m=meta["MN"][0], a=meta["MN"][1], b=meta["MN"][2], units=us)
    pot["bulge"] = gp.PowerLawCutoffPotential(m=meta["PSP"][0], alpha=meta["PSP"][1],
                                              r_c=meta["PSP"][2], units=us)
    return pot


class GalpyExternal:
    """galpy potentials in natural units with ro = 10 kpc and vo = code velocity.

    galpy has G = 1 in natural units, Odisseo has G = G_code in code units, so a galpy
    mass amplitude is G_code * M. Forces come out directly in code units.
    """

    def __init__(self, meta):
        from scipy.special import gamma
        from galpy.potential import (MiyamotoNagaiPotential, NFWPotential,
                                     PowerSphericalPotentialwCutoff)
        G = meta["G"]
        Mb, alpha, rc = meta["PSP"]
        # rho = amp (1/r)^alpha exp(-(r/rc)^2)  ->  M_tot = 2 pi amp rc^(3-alpha) Gamma(3/2 - alpha/2)
        amp_rho = Mb / (2 * np.pi * rc ** (3 - alpha) * gamma(1.5 - alpha / 2))
        self.pots = [NFWPotential(amp=G * meta["NFW"][0], a=meta["NFW"][1]),
                     MiyamotoNagaiPotential(amp=G * meta["MN"][0], a=meta["MN"][1], b=meta["MN"][2]),
                     PowerSphericalPotentialwCutoff(amp=G * amp_rho, alpha=alpha, rc=rc)]

    def acc(self, pos):
        from galpy.potential import evaluateRforces, evaluatezforces
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        R = np.hypot(x, y); phi = np.arctan2(y, x)
        FR = evaluateRforces(self.pots, R, z, phi=phi, use_physical=False)
        Fz = evaluatezforces(self.pots, R, z, phi=phi, use_physical=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            c, s = np.where(R > 0, x / R, 0.0), np.where(R > 0, y / R, 0.0)
        return np.stack([FR * c, FR * s, Fz], axis=1)


def direct_acc_numpy(pos, mass, eps, G):
    """Softened all-pairs acceleration in float64, blocked over rows."""
    n = len(mass)
    block = max(1, (1 << 22) // n)
    acc = np.empty_like(pos)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        d = pos[None, :, :] - pos[i0:i1, None, :]
        inv3 = (np.einsum("ijk,ijk->ij", d, d) + eps * eps) ** -1.5
        inv3[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
        acc[i0:i1] = G * np.einsum("ij,ijk->ik", inv3 * mass[None, :], d)
    return acc


def check_cpu_potentials() -> None:
    """gala and galpy accelerations vs Odisseo's at the setup's check points."""
    d = np.load(DATA / "potential_check.npz"); meta = json.loads(str(d["meta"]))
    us = gala_units()
    ag = gala_potential(meta).acceleration(d["pos"].T * us["length"]).decompose(us).value.T
    ap = GalpyExternal(meta).acc(d["pos"])
    rel = lambda x: np.max(np.linalg.norm(x - d["acc"], axis=1) / np.linalg.norm(d["acc"], axis=1))
    eg, ep = rel(ag), rel(ap)
    print(f"[check] external acceleration vs Odisseo: gala {eg:.1e}, galpy {ep:.1e} (max rel)", flush=True)
    if max(eg, ep) > 1e-10:
        raise SystemExit("FATAL: gala/galpy potential does not match Odisseo")


def run_cpu_code(a, code: str) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"   # numpy is imported already, but BLAS reads these lazily; einsum is single threaded
    if a.core is not None:
        os.sched_setaffinity(0, {a.core})
    import gala, galpy, scipy
    check_cpu_potentials()
    versions = json.dumps(dict(gala=gala.__version__, galpy=galpy.__version__, numpy=np.__version__,
                               scipy=scipy.__version__, python=platform.python_version()))
    for N in a.N:
        ic = load_ic(N); meta = ic["meta"]
        pos, vel, mass = ic["pos"], ic["vel"], ic["mass"]
        n_steps = a.n_steps or meta["n_steps"]
        for r in range(3 if N <= 1000 else 1):
            la0 = os.getloadavg()[0]
            row = dict(timestamp_utc=utcnow(), code=code, device="cpu", N=N, precision="fp64",
                       scheme=("DirectNBody+Leapfrog(sequential)" if code == "gala"
                               else "numpy-direct+galpy-ext+KDK"),
                       repeat=r, n_steps=n_steps, dt=meta["dt"], threads=1, loadavg_before=la0,
                       host=platform.node(), cpu_model=cpu_model(), versions=versions,
                       force_evals_per_step=1, n_force_evals=n_steps + 1)
            try:
                t0 = time.perf_counter()
                if code == "gala":
                    import gala.dynamics as gd
                    import gala.potential as gp
                    from gala.dynamics.nbody import DirectNBody
                    from gala.integrate import LeapfrogIntegrator
                    us = gala_units()
                    w0 = gd.PhaseSpacePosition(pos=pos.T * us["length"], vel=vel.T * us["length"] / us["time"])
                    pps = [gp.PlummerPotential(m=m, b=meta["softening"], units=us) for m in mass]
                    sim = DirectNBody(w0, pps, external_potential=gala_potential(meta), units=us, save_all=False)
                    setup = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    w = sim.integrate_orbit(Integrator=LeapfrogIntegrator, dt=meta["dt"], t1=0.0, n_steps=n_steps)
                    wall = time.perf_counter() - t0
                    x, v = (np.asarray(w.xyz.decompose(us).value).T, np.asarray(w.v_xyz.decompose(us).value).T)
                else:
                    ext = GalpyExternal(meta)
                    G, eps, dt = meta["G"], meta["softening"], meta["dt"]
                    setup = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    x, v = pos.copy(), vel.copy()
                    acc = direct_acc_numpy(x, mass, eps, G) + ext.acc(x)
                    for _ in range(n_steps):          # kick-drift-kick
                        v += 0.5 * dt * acc
                        x += dt * v
                        acc = direct_acc_numpy(x, mass, eps, G) + ext.acc(x)
                        v += 0.5 * dt * acc
                    wall = time.perf_counter() - t0
                ips = N * N * (n_steps + 1) / wall
                row.update(status="ok", wall_s=wall, setup_s=setup, interactions_per_s=ips,
                           loadavg_after=os.getloadavg()[0])
                if r == 0:
                    _save_final(a, code, N, x, v, ic, wall)
                print(f"[{code}] N={N:6d} rep={r} wall={wall:9.2f}s setup={setup:6.2f}s int/s={ips:.3e}",
                      flush=True)
            except Exception as e:
                row.update(status="error", error=repr(e)[:500])
                print(f"[{code}] N={N} FAILED: {e!r}", flush=True)
            if not a.n_steps:
                append_row(a.csv, row)


def cmd_run(a) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.code in ("odisseo", "odisseo_fp32", "odisseo_cpu"):
        run_odisseo(a, "cpu" if a.code == "odisseo_cpu" else "gpu", fp32=a.code == "odisseo_fp32")
    else:
        run_cpu_code(a, a.code)


# --------------------------------------------------------------------------- #
# Analysis (main venv, CPU only)                                               #
# --------------------------------------------------------------------------- #
def cmd_analyze(a) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- per (code, N) summary
    rows = [r for r in csv.DictReader(open(RESULTS / "stream_timings.csv")) if r["status"] == "ok"]
    groups = {}
    for r in rows:
        groups.setdefault((r["code"], int(r["N"])), []).append(r)
    ref = {N: np.median([float(r["wall_s"]) for r in rs]) for (c, N), rs in groups.items() if c == "odisseo"}

    def energy(pos, vel, mass, meta):
        _, phi = odisseo_external(pos)
        return (0.5 * np.sum(mass * np.sum(vel * vel, 1))
                + self_potential_energy(pos, mass, meta["softening"], meta["G"]) + float(np.sum(mass * phi)))

    summary = []
    for (code, N), rs in sorted(groups.items()):
        w = np.array([float(r["wall_s"]) for r in rs])
        f = RESULTS / f"final_{code}_N{N}.npz"
        dE = np.nan
        if f.exists():
            ic = load_ic(N)
            with np.load(f) as d:
                E1 = energy(d["pos"], d["vel"], d["mass"], ic["meta"])
            E0 = energy(ic["pos"], ic["vel"], ic["mass"], ic["meta"])
            dE = abs(E1 - E0) / abs(E0)
        summary.append(dict(code=code, N=N, median=np.median(w), min=w.min(), max=w.max(), count=len(w),
                            interactions_per_s=np.median([float(r["interactions_per_s"]) for r in rs]),
                            time_over_odisseo_gpu=np.median(w) / ref[N] if N in ref else np.nan, dE_E=dE))
    with open(RESULTS / "stream_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary[0])); wr.writeheader(); wr.writerows(summary)
    print(f"{'code':12s} {'N':>6s} {'median_s':>10s} {'x GPU':>8s} {'|dE/E|':>9s}")
    for s in summary:
        print(f"{s['code']:12s} {s['N']:6d} {s['median']:10.3f} {s['time_over_odisseo_gpu']:8.1f} {s['dE_E']:9.2e}")

    # ---- timing figure: common N grid (the N run by gala / galpy)
    grid = sorted({s["N"] for s in summary if s["code"] in ("gala", "galpy")})
    # same style as the docs scaling plot (notebooks/scalability, gpu_scaling_combined.png)
    fig, ax = plt.subplots(figsize=(12, 11))   # axes ~12x8 in, plus room for the legend above
    for code in CODES:
        ss = sorted([s for s in summary if s["code"] == code and s["N"] in grid], key=lambda s: s["N"])
        if not ss:
            continue
        N_ = [s["N"] for s in ss]; m = np.array([s["median"] for s in ss])
        err = [m - [s["min"] for s in ss], [s["max"] for s in ss] - m]
        ax.errorbar(N_, m, yerr=err, marker=MARKER[code], ls=LS[code], color=COLOR[code],
                    label=LABEL[code], capsize=5, ms=10, lw=2)
    gala0 = [s["median"] for s in summary if s["code"] == "gala" and s["N"] == grid[0]]
    if gala0:
        Ns = np.array(grid, float)
        ax.plot(Ns, gala0[0] * (Ns / Ns[0]) ** 2, ":", color="grey", lw=2, label=r"$\propto N^2$")
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("N particles", fontsize=30); ax.set_ylabel("Wall-clock time [s]", fontsize=30)
    ax.set_xticks(grid, [str(n) for n in grid]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=30)
    ax.grid(alpha=0.3, which="both")
    # legend above the axes, as in the conservation plot
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=20, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"stream_timing.{ext}", bbox_inches="tight", dpi=300)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="generate the IC files (main venv)")
    s.add_argument("--N", type=int, nargs="+", required=True)
    r = sub.add_parser("run", help="time one code")
    r.add_argument("--code", choices=CODES, required=True)
    r.add_argument("--N", type=int, nargs="+", required=True)
    r.add_argument("--core", type=int, default=None, help="pin to this CPU core (CPU runs)")
    r.add_argument("--batch-size", type=int, default=10000, help="DIRECT_ACC_LAXMAP batch size")
    r.add_argument("--gpu-timeout", type=int, default=None, help="autocvd wait [s]")
    r.add_argument("--csv", type=Path, default=RESULTS / "stream_timings.csv")
    r.add_argument("--n-steps", type=int, default=None,
                   help="testing only: fewer steps, nothing written to the CSV, final state to --out-dir")
    r.add_argument("--out-dir", default=str(HERE / "results" / "test"))
    sub.add_parser("analyze", help="summary CSV and plots (main venv)")
    a = ap.parse_args()
    {"setup": cmd_setup, "run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
