"""The default fixed-step FMM lane integrates self-gravity to second order.

Ported from the removed ``tests/test_fmm_leapfrog_order.py`` (Odisseo's own FMM
leapfrog loop, removed in 2026-11): the runner lane (``fmm_runner``), which replaced
it as the default, must satisfy what it pinned. Eight particles share one leaf, so
the FMM force is the direct sum to round-off and the end-state error is the
integrator's alone: halving ``dt`` must quarter it (velocity Verlet), and the run
must be velocity Verlet with the direct-sum Plummer field to round-off.

``integrate()`` with a config that sets no FMM knob, a float64 state (the fp64
force), on whatever backend runs the suite (a CPU takes the fused lane's CPU route).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from odisseo.integration_api import integrate
from odisseo.option_classes import FMM_ACC, SimulationConfig, SimulationParams

pytest.importorskip("jaccpot.production")

N = 8
EPS = 0.05
T_END = 1.0


def _initial_state():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(N, 3))
    vel = 0.3 * rng.normal(size=(N, 3))
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
    state = jnp.asarray(np.stack([pos, vel], axis=1), dtype=jnp.float64)
    mass = jnp.full((N,), 1.0 / N, dtype=jnp.float64)
    return state, mass


def _plummer_acc(pos, mass):
    """Direct-sum Plummer self-acceleration, G = 1."""
    diff = pos[None, :, :] - pos[:, None, :]
    r2 = jnp.sum(diff * diff, axis=-1) + EPS**2
    inv_r3 = jnp.where(jnp.eye(pos.shape[0], dtype=bool), 0.0, r2**-1.5)
    return jnp.sum(mass[None, :, None] * diff * inv_r3[..., None], axis=1)


def _velocity_verlet(state, mass, num_steps):
    dt = T_END / num_steps

    def body(carry, _):
        x, v, acc = carry
        x = x + v * dt + 0.5 * acc * dt**2
        acc_new = _plummer_acc(x, mass)
        v = v + 0.5 * (acc + acc_new) * dt
        return (x, v, acc_new), None

    x, v = state[:, 0], state[:, 1]
    (x, v, _), _ = jax.lax.scan(
        body, (x, v, _plummer_acc(x, mass)), None, length=num_steps
    )
    return jnp.stack([x, v], axis=1)


def test_the_runner_lane_is_second_order():
    state, mass = _initial_state()
    # velocity Verlet at 8192 steps: ~1e-6 of the 8-step error
    reference = np.asarray(jax.jit(_velocity_verlet, static_argnums=2)(state, mass, 8192))
    errors = []
    for n in (4, 8):
        cfg = SimulationConfig(
            N_particles=N,
            acceleration_scheme=FMM_ACC,
            num_timesteps=n,
            softening=EPS,
            softening_kernel="plummer",
        )
        out, lane = integrate(
            state, mass, cfg, SimulationParams(G=1.0, t_end=T_END), return_lane=True
        )
        assert lane == "fmm_runner"
        final = np.asarray(out)
        errors.append(np.max(np.abs(final - reference)))
    # the removed loop measured 3.92 (2.0 with a first-order closing kick)
    assert errors[0] / errors[1] > 3.5, f"errors {errors}"
    expected = np.asarray(_velocity_verlet(state, mass, 8))
    np.testing.assert_allclose(final, expected, rtol=0, atol=1e-12)
