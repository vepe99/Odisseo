import jax.numpy as jnp

from jaccpot.runtime._fmm_impl import _velocity_verlet_state_update
from odisseo.jaccpot_coupling import _large_n_environment_overrides
from odisseo.option_classes import SimulationConfig


def _canonical_config(**overrides):
    values = dict(
        N_particles=200_000,
        fmm_preset="large_n_gpu",
        fmm_tree_build_mode="static_radix",
    )
    values.update(overrides)
    return SimulationConfig(**values)


def test_strict_velocity_verlet_uses_endpoint_acceleration():
    state = jnp.asarray([[[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=jnp.float32)
    acceleration_current = -state[:, 0]
    dt = jnp.asarray(0.1, dtype=state.dtype)
    position_new = (
        state[:, 0]
        + state[:, 1] * dt
        + 0.5 * acceleration_current * dt**2
    )
    acceleration_new = -position_new

    actual = _velocity_verlet_state_update(
        state, acceleration_current, acceleration_new, dt
    )
    expected_velocity = state[:, 1] + 0.5 * (
        acceleration_current + acceleration_new
    ) * dt
    frozen_velocity = state[:, 1] + acceleration_current * dt

    assert jnp.allclose(actual[:, 0], position_new)
    assert jnp.allclose(actual[:, 1], expected_velocity)
    assert not jnp.allclose(actual[:, 1], frozen_velocity)


def test_canonical_static_radix_auto_cap_is_32():
    overrides = _large_n_environment_overrides(
        _canonical_config(), fmm_preset="large_n_gpu"
    )
    assert overrides["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"] == "32"


def test_explicit_static_radix_cap_is_not_increased():
    overrides = _large_n_environment_overrides(
        _canonical_config(fmm_large_n_static_target_blocks_max_per_leaf=16),
        fmm_preset="large_n_gpu",
    )
    assert overrides["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"] == "16"
