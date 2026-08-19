#!/usr/bin/env python3
"""Isolate whether fixed-step Hermite+FMM blow-up is the METHOD or the IC.

Same integrator (fixed-step Hermite-4) and same ICs, only the force backend
changes: exact direct-sum vs jaccpot FMM at several opening angles theta.
Energy drift is always measured with the EXACT direct-sum energy (a consistent
yardstick). Leapfrog is included as a jerk-free reference.

Logic:
  * Plummer sphere = a smooth, isotropic, self-gravitating EQUILIBRIUM. If
    direct-sum Hermite is stable there but FMM Hermite blows up, the IC is not
    the culprit -> it is the method (FMM force).
  * If tightening theta (more accurate, smoother FMM) removes the blow-up, the
    cause is the FMM approximation / force non-smoothness, not the integrator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED", "0")
os.environ.setdefault("ODISSEO_DISABLE_FMM_REFRESH_PREPARED_STATE", "1")

from autocvd import autocvd  # noqa: E402

autocvd(num_gpus=1)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from nornax.initial_conditions import sample_plummer_sphere  # noqa: E402
from odisseo import construct_initial_state  # noqa: E402
from odisseo.integration_api import integrate  # noqa: E402
from odisseo.option_classes import (  # noqa: E402
    DIRECT_ACC, FMM_ACC, HERMITE, LEAPFROG, NFW_POTENTIAL,
    NFWParams, SimulationConfig, SimulationParams,
)
from odisseo.utils import E_tot  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "scalability" / "ic_cache" / "odisseo_fixed_agama_ic_200k.npz"
N = 2048
SOFT = 0.05


def plummer_ic():
    pos, vel, mass = sample_plummer_sphere(
        jax.random.PRNGKey(0), N, total_mass=1.0, scale_radius=1.0)
    return construct_initial_state(pos, vel), mass, ()  # no external


def disk_ic():
    s = np.asarray(np.load(CACHE)["state0"], dtype=np.float64)
    idx = np.random.default_rng(0).choice(s.shape[0], N, replace=False)
    state0 = construct_initial_state(jnp.asarray(s[idx, 0]), jnp.asarray(s[idx, 1]))
    mass = jnp.full((N,), 6.0 / N, dtype=jnp.float64)
    return state0, mass, (NFW_POTENTIAL,)


def cfg(scheme, integrator, ext, nsteps, theta=0.6):
    return SimulationConfig(
        N_particles=N, integrator=integrator, acceleration_scheme=scheme,
        fixed_timestep=True, num_timesteps=nsteps, softening=SOFT,
        external_accelerations=ext, fmm_theta=theta, fmm_leaf_size=16,
        fmm_max_order=4, hermite_order=4, hermite_jerk_mode="fast_approx",
    )


def energy(state, mass, ext, params):
    c = SimulationConfig(N_particles=N, acceleration_scheme=DIRECT_ACC,
                         softening=SOFT, external_accelerations=ext)
    return float(jnp.sum(E_tot(state, mass, c, params)))


def run_case(state0, mass, ext, params, scheme, integrator, theta, nsteps, E0):
    c = cfg(scheme, integrator, ext, nsteps, theta)
    final = jax.block_until_ready(integrate(state0, mass, c, params))
    E1 = energy(final, mass, ext, params)
    r95 = float(jnp.percentile(jnp.linalg.norm(final[:, 0], axis=-1), 95))
    return {"rel_dE": abs((E1 - E0) / (abs(E0) + 1e-30)),
            "r95": r95, "finite": bool(jnp.all(jnp.isfinite(final)))}


def main():
    print("backend", jax.default_backend(), jax.devices())
    results = {}
    systems = [
        ("plummer", plummer_ic(), SimulationParams(G=1.0, t_end=4.0), 400),
        ("disk_in_halo", disk_ic(),
         SimulationParams(G=1.0, t_end=0.3, NFW_params=NFWParams(Mvir=100.0, r_s=2.0)),
         200),
    ]
    for sysname, (state0, mass, ext), params, nsteps in systems:
        E0 = energy(state0, mass, ext, params)
        print(f"\n===== {sysname} (N={N}, nsteps={nsteps}, E0={E0:.4e}) =====")
        cases = [
            ("direct", DIRECT_ACC, LEAPFROG, None),
            ("direct", DIRECT_ACC, HERMITE, None),
            ("fmm_theta0.6", FMM_ACC, LEAPFROG, 0.6),
            ("fmm_theta0.6", FMM_ACC, HERMITE, 0.6),
            ("fmm_theta0.3", FMM_ACC, HERMITE, 0.3),
            ("fmm_theta0.15", FMM_ACC, HERMITE, 0.15),
        ]
        results[sysname] = {}
        for label, scheme, integ, theta in cases:
            ig = "leapfrog" if integ == LEAPFROG else "hermite4"
            key = f"{label}/{ig}"
            try:
                m = run_case(state0, mass, ext, params, scheme, integ,
                             theta if theta else 0.6, nsteps, E0)
                results[sysname][key] = m
                print(f"  {key:>26}: dE/E={m['rel_dE']:.3e}  r95={m['r95']:.3f}  "
                      f"finite={m['finite']}")
            except Exception as e:
                results[sysname][key] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {key:>26}: ERROR {type(e).__name__}: {str(e)[:120]}")

    (HERE / "controlled_results.json").write_text(json.dumps(results, indent=2))
    print("\nWrote controlled_results.json")


if __name__ == "__main__":
    main()
