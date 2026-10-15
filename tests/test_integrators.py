import pytest
import jax.numpy as jnp
from odisseo.integrators import leapfrog, RungeKutta4, diffrax_solver
from odisseo.option_classes import (
    SimulationConfig, SimulationParams,
    DIRECT_ACC, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_LAXMAP,
    DOPRI5, TSIT5, SEMIIMPLICITEULER, REVERSIBLEHEUN, LEAPFROGMIDPOINT,
    NFW_POTENTIAL, PSP_POTENTIAL, MN_POTENTIAL, POINT_MASS, LOGARITHMIC_POTENTIAL, 
)



@pytest.fixture
def state():
    pos = jnp.array([[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]])
    vel = jnp.zeros_like(pos)
    return jnp.stack([pos, vel], axis=1)


@pytest.fixture
def mass():
    return jnp.array([1e1, 1e1, 1e1])


@pytest.fixture
def params():
    class DummyParams:
        G = 4.30091e-6
    return SimulationParams(G=DummyParams.G)


@pytest.mark.parametrize("integrator_func", [leapfrog, RungeKutta4])
def test_integrator_conserves_shape(state, mass, params, integrator_func):
    config = SimulationConfig(N_particles=3, acceleration_scheme=DIRECT_ACC)
    dt = jnp.array(1e-3)
    new_state = integrator_func(state, mass, dt, config, params)
    assert new_state.shape == state.shape


@pytest.mark.parametrize("scheme", [DIRECT_ACC, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_LAXMAP])
def test_integrators_with_different_schemes(state, mass, params, scheme):
    config = SimulationConfig(N_particles=3, acceleration_scheme=scheme)
    dt = jnp.array(1e-3)
    new_state = leapfrog(state, mass, dt, config, params)
    assert new_state.shape == state.shape


# diffrax_solver's SEMIIMPLICITEULER branch passes `saveat`, which only the other
# branch defines (odisseo/integrators.py). Binding it is not enough: that branch
# packs positions and velocities component-major, (3, N), while its velocity
# field `g` writes them into an (N, 3) state and returns (N, 3) accelerations, so
# it then fails to broadcast for N != 3, and at N = 3 it silently uses the
# transpose and returns a (3, 2, 3, 3) state. Known failure on
# jaccpot-integration at adca4f6; strict, so the fix has to remove the marker.
_SEMIIMPLICITEULER_UNBOUND_SAVEAT = pytest.param(
    SEMIIMPLICITEULER,
    marks=pytest.mark.xfail(
        strict=True,
        raises=UnboundLocalError,
        reason="the SEMIIMPLICITEULER branch uses an unbound `saveat`",
    ),
)


@pytest.mark.parametrize(
    "solver_option",
    [DOPRI5, TSIT5, _SEMIIMPLICITEULER_UNBOUND_SAVEAT, REVERSIBLEHEUN, LEAPFROGMIDPOINT],
)
def test_diffrax_solver_variants(state, mass, params, solver_option):
    config = SimulationConfig(N_particles=3, acceleration_scheme=DIRECT_ACC, diffrax_solver=solver_option)
    dt = jnp.array(1e-3)
    result = diffrax_solver(state, mass, dt, config, params)
    assert result.shape == state.shape


def test_leapfrog_step_symmetry(state, mass, params):
    """Leapfrog should preserve energy-like symmetry for small dt."""
    config = SimulationConfig(N_particles=3, acceleration_scheme=DIRECT_ACC)
    dt = jnp.array(1e-3)
    state1 = leapfrog(state, mass, dt, config, params)
    state2 = leapfrog(state1, mass, -dt, config, params)
    assert jnp.allclose(state, state2, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("integrator_func", [leapfrog, RungeKutta4, diffrax_solver])
def test_integrators_with_external_potential(state, mass, params, integrator_func):
    config = SimulationConfig(N_particles=3, acceleration_scheme=DIRECT_ACC, diffrax_solver=DOPRI5, external_accelerations=(NFW_POTENTIAL, PSP_POTENTIAL, MN_POTENTIAL, POINT_MASS, LOGARITHMIC_POTENTIAL, ))
    dt = jnp.array(1e-3)
    new_state = integrator_func(state, mass, dt, config, params)
    assert new_state.shape == state.shape


