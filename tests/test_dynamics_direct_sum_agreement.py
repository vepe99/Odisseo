"""Cross-checks that every direct-summation acceleration scheme agrees.

All five ``direct_acc*`` variants in :mod:`odisseo.dynamics` compute the same
physical quantity, so on identical inputs they must agree to float32 roundoff.
``direct_acc`` (the plain double ``vmap``) is used as the reference.
"""
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest

from odisseo.dynamics import (direct_acc, direct_acc_laxmap, direct_acc_matrix,
                              direct_acc_for_loop, direct_acc_sharding)
from odisseo.option_classes import SimulationConfig, SimulationParams

# Variants returning accelerations of shape (N, 3).
ACC_SCHEMES = [direct_acc, direct_acc_laxmap, direct_acc_matrix,
               direct_acc_for_loop, direct_acc_sharding]

# Variants returning an (acc, pot) pair when return_potential=True.
# direct_acc_for_loop is excluded: it returns the potential alone (see
# test_for_loop_potential_return_signature below).
POT_SCHEMES = [direct_acc, direct_acc_laxmap, direct_acc_matrix, direct_acc_sharding]

SOFTENING = 1e-3


def _system(N, seed=0, uniform_mass=True):
    """Random positions in [-1, 1]^3 with zero velocities."""
    key = jax.random.PRNGKey(seed)
    kpos, kmass = jax.random.split(key)
    pos = jax.random.uniform(kpos, (N, 3), minval=-1.0, maxval=1.0)
    state = jnp.stack([pos, jnp.zeros_like(pos)], axis=1)
    if uniform_mass:
        mass = jnp.ones(N)
    else:
        mass = jax.random.uniform(kmass, (N,), minval=0.25, maxval=4.0)
    config = SimulationConfig(N_particles=N, softening=SOFTENING)
    return state, mass, config, SimulationParams(G=1.0)


def _call(scheme, state, mass, config, params, **kwargs):
    # direct_acc_matrix and direct_acc_sharding are wrapped in
    # eqx.filter_jit(donate='all'), so hand each scheme its own buffers.
    return scheme(jnp.array(state), jnp.array(mass), config, params, **kwargs)


@pytest.mark.parametrize("scheme", ACC_SCHEMES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("N", [2, 4, 8, 9])
@pytest.mark.parametrize("uniform_mass", [True, False], ids=["uniform", "nonuniform"])
def test_acceleration_matches_direct_acc(scheme, N, uniform_mass):
    state, mass, config, params = _system(N, uniform_mass=uniform_mass)
    expected = direct_acc(state, mass, config, params)
    got = _call(scheme, state, mass, config, params)

    # Shape is asserted separately: broadcasting can hide a wrong-rank result
    # inside an allclose comparison.
    assert got.shape == (N, 3), f"{scheme.__name__} returned {got.shape}, expected {(N, 3)}"
    assert jnp.allclose(got, expected, atol=1e-4), (
        f"{scheme.__name__} max|diff| = {float(jnp.max(jnp.abs(got - expected))):.3e}")


@pytest.mark.parametrize("scheme", POT_SCHEMES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("N", [2, 4, 8, 9])
@pytest.mark.parametrize("uniform_mass", [True, False], ids=["uniform", "nonuniform"])
def test_potential_matches_direct_acc(scheme, N, uniform_mass):
    state, mass, config, params = _system(N, uniform_mass=uniform_mass)
    _, expected = direct_acc(state, mass, config, params, return_potential=True)
    got_acc, got_pot = _call(scheme, state, mass, config, params, return_potential=True)

    assert got_acc.shape == (N, 3), f"{scheme.__name__} acc {got_acc.shape} != {(N, 3)}"
    assert got_pot.shape == (N,), f"{scheme.__name__} pot {got_pot.shape} != {(N,)}"
    assert jnp.allclose(got_pot, expected, atol=1e-4), (
        f"{scheme.__name__} max|diff| = {float(jnp.max(jnp.abs(got_pot - expected))):.3e}")


def test_sharding_agrees_on_two_body(two_body_state):
    """The analytic two-body case, matching test_dynamics.test_full_pairwise_force."""
    state, mass = two_body_state
    config = SimulationConfig(N_particles=2, return_snapshots=False)
    params = SimulationParams(G=1)
    acc = direct_acc_sharding(state, mass, config, params, return_potential=False)
    assert jnp.allclose(acc, jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), atol=1e-5)


@pytest.mark.xfail(strict=True, reason="direct_acc_for_loop returns the potential "
                                       "alone, but odisseo.utils.E_pot unpacks an "
                                       "(acc, pot) pair from it")
def test_for_loop_potential_return_signature():
    state, mass, config, params = _system(8)
    acc, pot = direct_acc_for_loop(state, mass, config, params, return_potential=True)
    assert acc.shape == (8, 3)
    assert pot.shape == (8,)


# --- multi-device coverage -------------------------------------------------
# The number of XLA host devices is fixed when the CPU backend initialises, so a
# genuinely multi-shard run has to happen in a fresh interpreter.

_SUBPROCESS_CHECK = textwrap.dedent("""
    import jax, jax.numpy as jnp
    from odisseo.dynamics import direct_acc, direct_acc_sharding
    from odisseo.option_classes import SimulationConfig, SimulationParams

    n_dev = len(jax.devices())
    assert n_dev == 4, f"expected 4 forced host devices, got {n_dev}"

    # N divisible by the device count, and N that requires internal padding.
    for N in (4, 8, 16, 1, 3, 7, 9, 13):
        key = jax.random.PRNGKey(N)
        kpos, kmass = jax.random.split(key)
        pos = jax.random.uniform(kpos, (N, 3), minval=-1.0, maxval=1.0)
        state = jnp.stack([pos, jnp.zeros_like(pos)], axis=1)
        for mass in (jnp.ones(N),
                     jax.random.uniform(kmass, (N,), minval=0.25, maxval=4.0)):
            config = SimulationConfig(N_particles=N, softening=1e-3)
            params = SimulationParams(G=1.0)
            exp_acc, exp_pot = direct_acc(state, mass, config, params,
                                          return_potential=True)
            acc, pot = direct_acc_sharding(jnp.array(state), jnp.array(mass),
                                           config, params, return_potential=True)
            assert acc.shape == (N, 3), (N, acc.shape)
            assert pot.shape == (N,), (N, pot.shape)
            da = float(jnp.max(jnp.abs(acc - exp_acc)))
            dp = float(jnp.max(jnp.abs(pot - exp_pot)))
            assert da < 1e-4, f"N={N} acc max|diff|={da:.3e}"
            assert dp < 1e-4, f"N={N} pot max|diff|={dp:.3e}"
    print("OK")
""")


def test_sharding_across_forced_host_devices():
    """direct_acc_sharding must stay exact when the mesh really has >1 device."""
    env = dict(os.environ,
               JAX_PLATFORMS="cpu",
               XLA_FLAGS=os.environ.get("XLA_FLAGS", "")
                         + " --xla_force_host_platform_device_count=4")
    proc = subprocess.run([sys.executable, "-c", _SUBPROCESS_CHECK],
                          capture_output=True, text=True, env=env,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.returncode == 0, (
        f"multi-device sharding check failed:\n{proc.stdout}\n{proc.stderr}")
    assert "OK" in proc.stdout
