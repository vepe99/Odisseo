import pytest
import jax
import jax.numpy as jnp
from odisseo.initial_condition import (
    Plummer_sphere,
    ic_two_body,
    sample_position_on_sphere,
    sample_position_on_circle,
    inclined_position,
    inclined_circular_velocity,
)
from odisseo.option_classes import SimulationConfig, SimulationParams, PlummerParams
from types import SimpleNamespace


@pytest.fixture
def sim_config():
    return SimulationConfig(N_particles=100)


@pytest.fixture
def sim_params():
    return SimulationParams(
        G=1.0,
        Plummer_params=PlummerParams()
    )

def test_plummer_sphere_output_shapes(sim_config, sim_params):
    key = jax.random.PRNGKey(42)
    pos, vel, mass = Plummer_sphere(key, sim_config, sim_params)
    assert pos.shape == (sim_config.N_particles, 3)
    assert vel.shape == (sim_config.N_particles, 3)
    assert mass.shape == (sim_config.N_particles,)
    assert jnp.allclose(mass.sum(), 1.0, rtol=1e-2)

def test_ic_two_body():
    params = SimulationParams(G=1.0)
    pos, vel, mass = ic_two_body(1.0, 1.0, 1.0, 0.9, params)
    assert pos.shape == (2, 3)
    assert vel.shape == (2, 3)
    assert mass.shape == (2,)
    assert jnp.isclose(pos[1, 0], 1.0)

def test_sample_position_on_sphere():
    key = jax.random.PRNGKey(0)
    r = 2.0
    samples = sample_position_on_sphere(key, r, num_samples=1000)
    norms = jnp.linalg.norm(samples, axis=1)
    assert samples.shape == (1000, 3)
    assert jnp.allclose(norms, r, rtol=1e-2)

def test_sample_position_on_circle():
    key = jax.random.PRNGKey(1)
    r = 3.0
    samples = sample_position_on_circle(key, r, num_samples=1000)
    norms = jnp.linalg.norm(samples[:, :2], axis=1)
    assert samples.shape == (1000, 3)
    assert jnp.allclose(norms, r, rtol=1e-2)
    assert jnp.allclose(samples[:, 2], 0.0)

def test_inclined_position():
    pos = jnp.array([[1.0, 0.0, 0.0]])
    inc = jnp.array(jnp.pi / 2)
    rotated = inclined_position(pos, inc)
    assert rotated.shape == (1, 3)
    assert jnp.allclose(rotated[0, 0], 1.0)

def test_inclined_circular_velocity():
    pos = jnp.array([[1.0, 0.0, 0.0]])
    v_c = jnp.array([1.0])
    inc = jnp.array(jnp.pi / 2)
    vel = inclined_circular_velocity(pos, v_c, inc)
    assert vel.shape == (3,)
    assert jnp.allclose(jnp.linalg.norm(vel), 1.0, rtol=1e-2)
