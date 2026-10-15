"""The FMM leapfrog lane integrates self-gravity to second order.

``integrate_leapfrog_jaccpot_active`` runs its segment paths (every run outside
the strict large-N lane, and every history or profiled run) by scanning steps
with the self-acceleration of the segment start. Until 2026-10 that frozen
field also closed the segment's last half-kick, so even ``refresh_every=1``
gave ``v_{n+1} = v_n + dt a(x_n)``: first order, the end-state error halving
with dt instead of quartering. The closing half-kick now uses the
self-acceleration at the segment's end, which makes ``refresh_every=1`` velocity
Verlet, the strict lane's scheme.

Most tests swap jaccpot's solver for an exact direct sum, which isolates the
integrator from the multipole error and keeps every path cheap. One test runs
the real FMM at tiny N.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from odisseo import jaccpot_coupling
from odisseo.jaccpot_coupling import integrate_leapfrog_jaccpot_active
from odisseo.option_classes import SimulationConfig, SimulationParams

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


def _segmented_kdk(state, mass, num_steps, refresh_every):
    """The scheme the lane implements, written out step by step.

    Both half-kicks inside a segment use the self-field frozen at the segment
    start, except the segment's last closing half-kick, which uses the field at
    the segment's end -- the next segment's frozen field.
    """
    dt = T_END / num_steps
    x, v = state[:, 0], state[:, 1]
    acc_frozen = _plummer_acc(x, mass)
    for step in range(num_steps):
        x_new = x + v * dt + 0.5 * acc_frozen * dt**2
        ends_segment = (step + 1) % refresh_every == 0 or step + 1 == num_steps
        acc_close = _plummer_acc(x_new, mass) if ends_segment else acc_frozen
        v = v + 0.5 * (acc_frozen + acc_close) * dt
        x = x_new
        acc_frozen = acc_close
    return jnp.stack([x, v], axis=1)


@jax.jit
def _velocity_verlet_reference(state, mass):
    """Velocity Verlet at 8192 steps: ~1e-6 of the 8-step error."""
    num_steps = 8192
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


class _DirectSumSolver:
    """Exact stand-in for jaccpot's solver: prepare keeps the sources."""

    def prepare_state(self, positions, masses, **_):
        return jnp.asarray(positions), jnp.asarray(masses)

    def refresh_prepared_state(self, prepared, positions, masses, **_):
        # jaccpot raises this outside the large-N lane; the lane tolerates it.
        raise NotImplementedError

    def evaluate_prepared_state(
        self, prepared, target_indices=None, return_potential=False
    ):
        acc = _plummer_acc(*prepared)
        return acc if target_indices is None else acc[target_indices]


@pytest.fixture
def direct_sum_solver(monkeypatch):
    monkeypatch.setattr(
        jaccpot_coupling,
        "_build_fmm_solver",
        lambda *args, **kwargs: _DirectSumSolver(),
    )


def _config():
    return SimulationConfig(N_particles=N, softening=EPS, softening_kernel="plummer")


# Every non-strict path of the lane, by the inputs that select it.
PATHS = {
    "final_state": {},
    "history": {"return_history": True},
    "profiled": {"timing_stats": {}},
    "active_schedule": {"active_schedule": True},
    "active_fn": {"active_indices_fn": lambda step, state, mass: jnp.arange(N)},
    "refresh_after_drift": {"refresh_after_position_update": True},
}


def _run(state, mass, num_steps, refresh_every, path):
    kwargs = dict(PATHS[path])
    if kwargs.pop("active_schedule", False):
        kwargs["active_indices_schedule"] = jnp.tile(jnp.arange(N), (num_steps, 1))
    if "timing_stats" in kwargs:
        kwargs["timing_stats"] = {}
    out = integrate_leapfrog_jaccpot_active(
        state,
        mass,
        _config(),
        SimulationParams(G=1.0, t_end=T_END),
        num_steps=num_steps,
        refresh_every=refresh_every,
        leaf_size=16,
        **kwargs,
    )
    out = np.asarray(out)
    if kwargs.get("return_history"):
        assert out.shape == (num_steps, N, 2, 3)
        return out[-1]
    return out


@pytest.mark.parametrize("path", sorted(PATHS))
def test_refresh_every_one_is_second_order(direct_sum_solver, path):
    state, mass = _initial_state()
    reference = np.asarray(_velocity_verlet_reference(state, mass))
    errors = [
        np.max(np.abs(_run(state, mass, n, 1, path) - reference)) for n in (8, 16, 32)
    ]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    # A first-order closing kick gives ratios of about 1.9 here.
    assert min(ratios) > 3.5, f"errors {errors}, ratios {ratios}"


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("refresh_every", [1, 3])
def test_segment_boundaries_close_with_endpoint_field(
    direct_sum_solver, path, refresh_every
):
    # 8 steps at refresh_every=3: two full segments and a tail of two.
    state, mass = _initial_state()
    got = _run(state, mass, 8, refresh_every, path)
    if path == "refresh_after_drift" and refresh_every > 1:
        # Refreshing after every drift closes every step with the endpoint field.
        expected = _segmented_kdk(state, mass, 8, 1)
    else:
        expected = _segmented_kdk(state, mass, 8, refresh_every)
    np.testing.assert_allclose(got, np.asarray(expected), rtol=0, atol=1e-12)


def test_history_run_with_real_fmm_is_second_order():
    # All 8 particles share one leaf, so the FMM force is the direct sum to
    # round-off and the end-state error is the integrator's alone.
    state, mass = _initial_state()
    reference = np.asarray(_velocity_verlet_reference(state, mass))
    errors = []
    for n in (4, 8):
        out = integrate_leapfrog_jaccpot_active(
            state,
            mass,
            _config(),
            SimulationParams(G=1.0, t_end=T_END),
            num_steps=n,
            refresh_every=1,
            leaf_size=16,
            return_history=True,
        )
        final = np.asarray(out)[-1]
        errors.append(np.max(np.abs(final - reference)))
    # 3.92 measured; 2.0 with a first-order closing kick.
    assert errors[0] / errors[1] > 3.5, f"errors {errors}"
    expected = _segmented_kdk(state, mass, 8, 1)
    np.testing.assert_allclose(final, np.asarray(expected), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# padded active-index schedules
# --------------------------------------------------------------------------


def test_masked_scatter_keeps_the_update_of_an_active_particle_zero():
    # Masked-out rows used to be aimed at row 0 carrying row 0's old value, so a
    # row with particle 0 active and padding after it wrote row 0 twice, and the
    # scatter kept the old value (last writer on CPU; most cases on GPU).
    from odisseo.jaccpot_coupling import _scatter_masked_vectors

    base = jnp.zeros((4, 3))
    indices = jnp.asarray([0, 2, 0, 0], dtype=jnp.int32)
    mask = jnp.asarray([True, True, False, False])
    values = jnp.asarray([[1.0] * 3, [2.0] * 3, [9.0] * 3, [9.0] * 3])
    out = np.asarray(_scatter_masked_vectors(base, indices, values, mask))
    np.testing.assert_array_equal(out[:, 0], [1.0, 0.0, 2.0, 0.0])


def test_padding_an_active_schedule_changes_nothing(direct_sum_solver):
    # Every particle active, plus two masked-out padding slots per row.
    state, mass = _initial_state()
    num_steps = 4
    params = SimulationParams(G=1.0, t_end=T_END)
    rows = jnp.tile(jnp.arange(N), (num_steps, 1))
    padded = jnp.concatenate([rows, jnp.zeros((num_steps, 2), jnp.int32)], axis=1)
    mask = jnp.concatenate(
        [jnp.ones((num_steps, N), bool), jnp.zeros((num_steps, 2), bool)], axis=1
    )
    common = dict(num_steps=num_steps, refresh_every=1, leaf_size=16)
    unpadded = integrate_leapfrog_jaccpot_active(
        state, mass, _config(), params, active_indices_schedule=rows, **common
    )
    got = integrate_leapfrog_jaccpot_active(
        state,
        mass,
        _config(),
        params,
        active_indices_schedule=padded,
        active_mask_schedule=mask,
        **common,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(unpadded))
