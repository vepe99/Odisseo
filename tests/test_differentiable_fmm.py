"""Gradients through the FMM lane, including external-potential parameters.

The forward FMM coupler cannot be differentiated (static ``params``, prebaked
prepared-state expansions), so :mod:`odisseo.differentiable` adds a lane that
re-evaluates self-gravity from the live positions at frozen topology. These tests
pin the properties that make the lane trustworthy:

* the gradient w.r.t. an external-potential parameter is **exact** for the frozen
  function -- it matches finite differences of the same frozen plan,
* it agrees with the direct-sum lane's gradient to the FMM's force accuracy, and
* the FMM's own sensitivity is genuinely present. The failure mode of a half-wired
  gradient seam is a silently *zero* self-gravity contribution, so the mass
  gradient (masses enter through self-gravity and nothing else) is checked against
  finite differences directly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from odisseo.differentiable import (
    DifferentiableFMMPlan,
    differentiable_fmm_self_acceleration,
    integrate_diffrax_differentiable,
    integrate_leapfrog_differentiable,
    prepare_differentiable_fmm,
    topology_drift,
)
from odisseo.integration_api import integrate
from odisseo.option_classes import (
    DIRECT_ACC,
    FMM_ACC,
    LEAPFROG,
    NFW_POTENTIAL,
    NFWParams,
    SimulationConfig,
    SimulationParams,
)
from odisseo.time_integration import time_integration

jax.config.update("jax_enable_x64", True)

N_PARTICLES = 96
NUM_STEPS = 4
MVIR0 = 100.0
RS0 = 15.3
T_END = 0.4

# Masses are chosen so self-gravity DOMINATES the external field over the
# integration: total mass ~9.6 within ~2 kpc gives |a_self| ~ 2 against the NFW's
# ~0.2. A weak-self-gravity setup would pass these tests even with a broken FMM
# gradient, because the external term alone would carry the sensitivity.
PARTICLE_MASS = 0.1


def _jaccpot_has_grad_seam() -> bool:
    try:
        from jaccpot import FastMultipoleMethod
    except ImportError:
        return False
    return callable(getattr(FastMultipoleMethod, "differentiable_accelerations", None))


pytestmark = pytest.mark.skipif(
    not _jaccpot_has_grad_seam(),
    reason="installed jaccpot has no FastMultipoleMethod.differentiable_accelerations",
)


@pytest.fixture(scope="module")
def initial_conditions():
    key = jax.random.PRNGKey(0)
    key_pos, key_vel = jax.random.split(key)
    pos = jax.random.normal(key_pos, (N_PARTICLES, 3)) * 2.0 + jnp.array(
        [10.0, 0.0, 0.0]
    )
    vel = jax.random.normal(key_vel, (N_PARTICLES, 3)) * 0.05
    state = jnp.stack([pos, vel], axis=1)
    mass = jnp.full((N_PARTICLES,), PARTICLE_MASS)
    return state, mass


def make_params(mvir=MVIR0, r_s=RS0, t_end=T_END):
    return SimulationParams(
        G=1.0,
        t_end=t_end,
        NFW_params=NFWParams(Mvir=mvir, r_s=r_s),
    )


def make_config(scheme=FMM_ACC, *, differentiable=False, snapshots=False):
    return SimulationConfig(
        N_particles=N_PARTICLES,
        num_timesteps=NUM_STEPS,
        fixed_timestep=True,
        return_snapshots=snapshots,
        num_snapshots=3,
        integrator=LEAPFROG,
        acceleration_scheme=scheme,
        softening=1e-3,
        external_accelerations=(NFW_POTENTIAL,),
        fmm_leaf_size=16,
        fmm_max_order=6,
        fmm_theta=0.4,
        fmm_basis="real",
        fmm_tree_build_mode="static_radix",
        fmm_auto_large_n_profile=False,
        fmm_use_pallas=False,
        fmm_differentiable=differentiable,
    )


@pytest.fixture(scope="module")
def plan(initial_conditions):
    state, mass = initial_conditions
    return prepare_differentiable_fmm(
        state, mass, make_config(differentiable=True), make_params()
    )


def _final_positions(plan, state, mass, params):
    config = make_config(differentiable=True)
    return integrate_leapfrog_differentiable(
        state, mass, config, params, plan=plan, num_steps=NUM_STEPS
    )[:, 0]


def _loss(plan, state, mass, params):
    """Sum of squared final positions -- a scalar that touches every particle."""
    return jnp.sum(_final_positions(plan, state, mass, params) ** 2)


def _central_difference(fn, x0, rel_step=1e-6):
    h = abs(x0) * rel_step
    return (fn(x0 + h) - fn(x0 - h)) / (2.0 * h)


# --------------------------------------------------------------------------
# plan construction
# --------------------------------------------------------------------------


def test_plan_is_built_from_concrete_inputs(plan):
    assert isinstance(plan, DifferentiableFMMPlan)
    assert plan.n_particles == N_PARTICLES
    assert callable(getattr(plan.solver, "differentiable_accelerations", None))


def test_plan_build_rejects_traced_state(initial_conditions):
    """The tree build is host-side; a tracer must raise, not silently degrade."""
    state, mass = initial_conditions
    config = make_config(differentiable=True)

    def build_inside_grad(scale):
        prepare_differentiable_fmm(state * scale, mass, config, make_params())
        return 0.0

    with pytest.raises(NotImplementedError, match="CONCRETE state/mass"):
        jax.grad(build_inside_grad)(1.0)


def test_integrator_without_plan_rejects_traced_inputs(initial_conditions):
    state, mass = initial_conditions
    config = make_config(differentiable=True)

    with pytest.raises(NotImplementedError, match="prepare_differentiable_fmm"):
        jax.grad(
            lambda s: jnp.sum(
                integrate_leapfrog_differentiable(
                    s, mass, config, make_params(), num_steps=1
                )
            )
        )(state)


# --------------------------------------------------------------------------
# forward agreement: the gradient is only useful if the force is right
# --------------------------------------------------------------------------


def test_self_acceleration_matches_direct_sum(initial_conditions, plan):
    """Fixed-topology FMM self-gravity must reproduce the exact O(N^2) sum."""
    from odisseo.dynamics import direct_acc

    state, mass = initial_conditions
    acc_fmm = differentiable_fmm_self_acceleration(plan, state[:, 0, :], mass)
    acc_direct = direct_acc(state, mass, make_config(DIRECT_ACC), make_params())

    scale = float(jnp.max(jnp.linalg.norm(acc_direct, axis=1)))
    err = float(jnp.max(jnp.linalg.norm(acc_fmm - acc_direct, axis=1))) / scale
    assert err < 1e-3, f"FMM self-gravity deviates from the direct sum by {err:.2e}"


def test_self_gravity_dominates_the_test_problem(initial_conditions, plan):
    """Guard the premise: a weak self-gravity setup would not test the FMM grad."""
    from odisseo.potentials import combined_external_acceleration_vmpa_switch

    state, mass = initial_conditions
    acc_self = differentiable_fmm_self_acceleration(plan, state[:, 0, :], mass)
    acc_ext = combined_external_acceleration_vmpa_switch(
        state, make_config(differentiable=True), make_params()
    )
    assert float(jnp.linalg.norm(acc_self)) > float(jnp.linalg.norm(acc_ext))


def test_differentiable_lane_forward_matches_direct_sum_lane(initial_conditions, plan):
    state, mass = initial_conditions
    params = make_params()

    final_diff = integrate_leapfrog_differentiable(
        state, mass, make_config(differentiable=True), params, plan=plan
    )
    final_direct = time_integration(state, mass, make_config(DIRECT_ACC), params)

    scale = float(jnp.max(jnp.linalg.norm(final_direct[:, 0], axis=1)))
    dev = float(jnp.max(jnp.linalg.norm(final_diff[:, 0] - final_direct[:, 0], axis=1)))
    assert dev / scale < 1e-4


# --------------------------------------------------------------------------
# the actual question: gradients onto external-potential parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, x0, replace",
    [
        ("Mvir", MVIR0, lambda x: make_params(mvir=x)),
        ("r_s", RS0, lambda x: make_params(r_s=x)),
        ("t_end", T_END, lambda x: make_params(t_end=x)),
    ],
)
def test_external_parameter_gradient_matches_finite_differences(
    initial_conditions, plan, name, x0, replace
):
    """AD must equal FD *of the same frozen plan* to near machine precision.

    Both sides differentiate the identical fixed-topology function, so this is a
    correctness test with no FMM-accuracy slack: a mis-wired reverse rule shows up
    immediately.
    """
    state, mass = initial_conditions

    def loss(x):
        return _loss(plan, state, mass, replace(x))

    grad_ad = float(jax.grad(loss)(x0))
    grad_fd = float(_central_difference(loss, x0))

    assert np.isfinite(grad_ad)
    assert abs(grad_ad) > 0.0, f"d loss/d {name} came back zero"
    rel = abs(grad_ad - grad_fd) / abs(grad_fd)
    assert rel < 1e-5, f"d loss/d {name}: AD {grad_ad:.8e} vs FD {grad_fd:.8e}"


def test_external_parameter_gradient_matches_direct_sum_lane(initial_conditions, plan):
    """The FMM gradient must agree with the exact-force lane's gradient.

    Tolerance is the FMM's own force accuracy, not machine precision: the FMM
    gradient is an exact gradient *of the FMM force*.
    """
    state, mass = initial_conditions

    grad_fmm = float(
        jax.grad(lambda m: _loss(plan, state, mass, make_params(mvir=m)))(MVIR0)
    )
    grad_direct = float(
        jax.grad(
            lambda m: jnp.sum(
                time_integration(
                    state, mass, make_config(DIRECT_ACC), make_params(mvir=m)
                )[:, 0]
                ** 2
            )
        )(MVIR0)
    )

    rel = abs(grad_fmm - grad_direct) / abs(grad_direct)
    assert rel < 1e-3, f"FMM {grad_fmm:.8e} vs direct-sum {grad_direct:.8e}"


def test_mass_gradient_matches_finite_differences(initial_conditions, plan):
    """Masses enter the dynamics ONLY through self-gravity.

    This is the sharpest available probe of the FMM reverse pass: the
    characteristic failure of a half-wired gradient seam is a self-gravity term
    with no sensitivity, and that failure makes this derivative exactly zero.
    """
    state, mass = initial_conditions
    index = 7

    def loss(mass_i):
        return _loss(plan, state, mass.at[index].set(mass_i), make_params())

    grad_ad = float(jax.grad(loss)(float(mass[index])))
    grad_fd = float(_central_difference(loss, float(mass[index])))

    assert abs(grad_ad) > 0.0, "self-gravity carries no mass sensitivity"
    assert abs(grad_ad - grad_fd) / abs(grad_fd) < 1e-5


def test_position_gradient_matches_finite_differences(initial_conditions, plan):
    """Probes d a_self / d x through the frozen topology, component by component."""
    state, mass = initial_conditions
    index, component = 11, 0
    x0 = float(state[index, 0, component])

    def loss(x):
        return _loss(
            plan, state.at[index, 0, component].set(x), mass, make_params()
        )

    grad_ad = float(jax.grad(loss)(x0))
    grad_fd = float(_central_difference(loss, x0))

    assert abs(grad_ad) > 0.0
    assert abs(grad_ad - grad_fd) / abs(grad_fd) < 1e-5


def test_frozen_self_gravity_changes_the_parameter_gradient(initial_conditions, plan):
    """The self-gravity *response* term contributes to d loss / d Mvir.

    Freezing self-gravity with ``stop_gradient`` is the gradient a path with a
    dead FMM reverse would return. The difference is small in absolute terms --
    the response enters as (external perturbation -> position shift -> changed
    self-gravity), which is higher order in ``dt``, and measures ~1.4e-4 relative
    over this four-step window -- but it is four orders of magnitude above the
    ~3e-8 AD-vs-FD agreement of the same run, so it is signal, not noise.
    """
    state, mass = initial_conditions
    config = make_config(differentiable=True)

    def loss_frozen_self(mvir):
        from odisseo.potentials import combined_external_acceleration_vmpa_switch

        params = make_params(mvir=mvir)
        dt = params.t_end / NUM_STEPS
        state_curr = state

        def acc_of(s):
            acc_self = jax.lax.stop_gradient(
                differentiable_fmm_self_acceleration(plan, s[:, 0, :], mass)
            )
            return acc_self + combined_external_acceleration_vmpa_switch(
                s, config, params
            )

        acc = acc_of(state_curr)
        for _ in range(NUM_STEPS):
            pos_new = (
                state_curr[:, 0] + state_curr[:, 1] * dt + 0.5 * acc * (dt**2)
            )
            state_curr = state_curr.at[:, 0].set(pos_new)
            acc_new = acc_of(state_curr)
            state_curr = state_curr.at[:, 1].set(
                state_curr[:, 1] + 0.5 * (acc + acc_new) * dt
            )
            acc = acc_new
        return jnp.sum(state_curr[:, 0] ** 2)

    grad_full = float(
        jax.grad(lambda m: _loss(plan, state, mass, make_params(mvir=m)))(MVIR0)
    )
    grad_frozen = float(jax.grad(loss_frozen_self)(MVIR0))

    rel = abs(grad_full - grad_frozen) / abs(grad_full)
    assert rel > 1e-5, (
        "freezing the FMM self-gravity response changed the parameter gradient by "
        f"only {rel:.2e}, at the level of the AD/FD floor; the test problem does "
        "not exercise the FMM reverse pass"
    )


def test_gradients_w_r_t_initial_state_and_masses(initial_conditions, plan):
    state, mass = initial_conditions

    grad_state, grad_mass = jax.grad(
        lambda s, m: _loss(plan, s, m, make_params()), argnums=(0, 1)
    )(state, mass)

    assert bool(jnp.all(jnp.isfinite(grad_state)))
    assert bool(jnp.all(jnp.isfinite(grad_mass)))
    # Positions and velocities both move the final positions; masses enter
    # through self-gravity only, which is why this is the term that catches a
    # dropped FMM mass-sensitivity.
    assert float(jnp.linalg.norm(grad_state[:, 0])) > 0.0
    assert float(jnp.linalg.norm(grad_state[:, 1])) > 0.0
    assert float(jnp.linalg.norm(grad_mass)) > 0.0


def test_history_gradient_flows_through_intermediate_states(initial_conditions, plan):
    """A snapshot-based loss (as in stream fitting) must differentiate too."""
    state, mass = initial_conditions

    def loss(mvir):
        history = integrate_leapfrog_differentiable(
            state,
            mass,
            make_config(differentiable=True),
            make_params(mvir=mvir),
            plan=plan,
            return_history=True,
        )
        assert history.shape == (NUM_STEPS + 1, N_PARTICLES, 2, 3)
        return jnp.sum(history[:, :, 0, :] ** 2)

    grad_ad = float(jax.grad(loss)(MVIR0))
    grad_fd = float(_central_difference(loss, MVIR0))
    assert abs(grad_ad - grad_fd) / abs(grad_fd) < 1e-5


# --------------------------------------------------------------------------
# integrate() wiring
# --------------------------------------------------------------------------


def test_integrate_routes_to_the_differentiable_lane(initial_conditions, plan):
    state, mass = initial_conditions
    params = make_params()

    via_api = integrate(
        state, mass, make_config(differentiable=True), params, fmm_plan=plan
    )
    direct = integrate_leapfrog_differentiable(
        state, mass, make_config(differentiable=True), params, plan=plan
    )
    assert bool(jnp.array_equal(via_api, direct))

    grad_api = float(
        jax.grad(
            lambda m: jnp.sum(
                integrate(
                    state,
                    mass,
                    make_config(differentiable=True),
                    make_params(mvir=m),
                    fmm_plan=plan,
                )[:, 0]
                ** 2
            )
        )(MVIR0)
    )
    grad_fd = float(
        _central_difference(lambda m: _loss(plan, state, mass, make_params(mvir=m)), MVIR0)
    )
    assert abs(grad_api - grad_fd) / abs(grad_fd) < 1e-5


def test_integrate_snapshots_on_the_differentiable_lane(initial_conditions, plan):
    state, mass = initial_conditions
    config = make_config(differentiable=True, snapshots=True)

    snapshots = integrate(state, mass, config, make_params(), fmm_plan=plan)
    assert snapshots.states.shape == (3, N_PARTICLES, 2, 3)
    assert snapshots.times.shape == (3,)


def test_forward_fmm_lane_rejects_traced_params(initial_conditions):
    """Without the flag the forward lane must explain itself, not fail on hashing."""
    state, mass = initial_conditions

    with pytest.raises(NotImplementedError, match="fmm_differentiable=True"):
        jax.grad(
            lambda m: jnp.sum(
                integrate(state, mass, make_config(FMM_ACC), make_params(mvir=m))[:, 0]
                ** 2
            )
        )(MVIR0)


def test_active_particle_scheduling_is_rejected(initial_conditions, plan):
    state, mass = initial_conditions
    with pytest.raises(NotImplementedError, match="active-particle scheduling"):
        integrate(
            state,
            mass,
            make_config(differentiable=True),
            make_params(),
            fmm_plan=plan,
            active_indices_schedule=jnp.zeros((NUM_STEPS, 4), dtype=jnp.int32),
        )


# --------------------------------------------------------------------------
# adaptive lane and diagnostics
# --------------------------------------------------------------------------


def test_diffrax_lane_gradient_matches_finite_differences(initial_conditions, plan):
    state, mass = initial_conditions
    config = make_config(differentiable=True)._replace(fixed_timestep=False)

    def loss(mvir):
        final = integrate_diffrax_differentiable(
            state, mass, config, make_params(mvir=mvir), plan=plan
        )
        return jnp.sum(final[:, 0] ** 2)

    grad_ad = float(jax.grad(loss)(MVIR0))
    grad_fd = float(_central_difference(loss, MVIR0, rel_step=1e-5))
    assert np.isfinite(grad_ad)
    # Adaptive stepping makes the FD reference noisier than the fixed-step lane:
    # the step sequence itself can change between the two perturbed solves.
    assert abs(grad_ad - grad_fd) / abs(grad_fd) < 1e-3


def test_zero_steps_returns_the_initial_state(initial_conditions, plan):
    """A zero-step window must not divide by ``num_steps`` to get ``dt``."""
    state, mass = initial_conditions
    config = make_config(differentiable=True)

    final = integrate_leapfrog_differentiable(
        state, mass, config, make_params(), plan=plan, num_steps=0
    )
    assert bool(jnp.array_equal(final, state))

    history = integrate_leapfrog_differentiable(
        state, mass, config, make_params(), plan=plan, num_steps=0, return_history=True
    )
    assert history.shape == (1, N_PARTICLES, 2, 3)


def test_topology_drift_reports_small_drift_for_a_short_window(
    initial_conditions, plan
):
    state, mass = initial_conditions
    final = integrate_leapfrog_differentiable(
        state, mass, make_config(differentiable=True), make_params(), plan=plan
    )

    drift = topology_drift(plan, final[:, 0])
    assert drift["max_displacement"] > 0.0
    assert drift["rms_displacement"] <= drift["max_displacement"]
    # A four-step window must not move particles anywhere near a cell width;
    # if it does, the fixed-topology contract is being stretched.
    assert drift["max_displacement_over_leaf_extent"] < 1.0


def test_topology_drift_rejects_tracers(plan, initial_conditions):
    state, _ = initial_conditions
    with pytest.raises(NotImplementedError, match="concrete positions"):
        jax.grad(lambda s: topology_drift(plan, s)["max_displacement"])(state[:, 0, :])
