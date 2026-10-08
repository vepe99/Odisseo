"""The softening kernels of ODISSEO's direct sums.

``odisseo.softening`` mirrors ``jaccpot.softening`` (ODISSEO does not depend on
jaccpot). Pinned here: the mirror agrees with jaccpot to round-off when jaccpot is
installed; every direct-sum variant evaluates the configured kernel (against a
NumPy loop, with pairs inside the support); Plummer keeps its historical values;
the compact kernels are Newtonian past the support.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from odisseo.dynamics import (
    direct_acc,
    direct_acc_for_loop,
    direct_acc_laxmap,
    direct_acc_matrix,
)
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.softening import (
    DEFAULT_SOFTENING_KERNEL,
    SOFTENING_KERNELS,
    softened_inverse_powers,
    support_radius,
)

_EPS = 0.3


def test_default_is_ferrers3_and_the_config_carries_it():
    assert DEFAULT_SOFTENING_KERNEL == "ferrers3"
    assert SimulationConfig().softening_kernel == "ferrers3"
    assert support_radius("ferrers3", 1.0) == 315.0 / 128.0
    assert support_radius("wendland_c2", 1.0) == 3.0
    assert support_radius("plummer", 1.0) == 0.0


@pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
def test_mirror_agrees_with_jaccpot(kernel):
    js = pytest.importorskip("jaccpot.softening")
    r2 = np.linspace(0.0, 3.0, 400) ** 2
    g, psi = softened_inverse_powers(r2, _EPS, kernel, xp=np)
    jg, jpsi, _ = js.pair_factors(
        r2, js.softening_params_np(kernel, _EPS), kernel, potential=True, xp=np
    )
    np.testing.assert_allclose(g, jg, rtol=1e-13)
    np.testing.assert_allclose(psi, jpsi, rtol=1e-13)


@pytest.mark.parametrize("kernel", ("ferrers3", "wendland_c2"))
def test_compact_kernels_are_newtonian_past_the_support(kernel):
    h = support_radius(kernel, _EPS)
    r = np.linspace(1.0001 * h, 5 * h, 50)
    g, psi = softened_inverse_powers(r * r, _EPS, kernel, xp=np)
    np.testing.assert_allclose(g, r**-3, rtol=1e-14)
    np.testing.assert_allclose(psi, 1.0 / r, rtol=1e-14)
    # and the central potential is Plummer's -G m / eps
    np.testing.assert_allclose(softened_inverse_powers(0.0, _EPS, kernel, xp=np)[1], 1 / _EPS)


def _loop(pos, mass, kernel, G):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    pot = np.zeros(n)
    for i in range(n):
        d = pos[i] - np.delete(pos, i, axis=0)
        m = np.delete(mass, i)
        g, psi = softened_inverse_powers(np.sum(d * d, axis=1), _EPS, kernel, xp=np)
        acc[i] = -G * np.sum((m * g)[:, None] * d, axis=0)
        pot[i] = -G * np.sum(m * psi)
    return acc, pot


@pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
@pytest.mark.parametrize(
    "force_func", [direct_acc, direct_acc_matrix, direct_acc_laxmap, direct_acc_for_loop]
)
def test_direct_sums_evaluate_the_configured_kernel(kernel, force_func):
    rng = np.random.default_rng(3)
    n = 24
    pos = rng.standard_normal((n, 3)) * 0.4  # many pairs inside h ~ 0.74 / 0.9
    mass = rng.uniform(0.5, 1.5, n)
    state = jnp.stack([jnp.asarray(pos), jnp.zeros((n, 3))], axis=1)
    config = SimulationConfig(N_particles=n, softening=_EPS, softening_kernel=kernel)
    params = SimulationParams(G=1.3)
    want_acc, want_pot = _loop(pos, mass, kernel, 1.3)
    acc = np.asarray(force_func(state, jnp.asarray(mass), config, params, return_potential=False))
    np.testing.assert_allclose(acc, want_acc, rtol=1e-5, atol=1e-6 * np.abs(want_acc).max())
    if force_func is not direct_acc_for_loop:  # its potential keeps the self term
        _, pot = force_func(state, jnp.asarray(mass), config, params, return_potential=True)
        np.testing.assert_allclose(np.asarray(pot), want_pot, rtol=1e-5)


def test_plummer_branch_is_the_historical_formula():
    rng = np.random.default_rng(5)
    n = 16
    pos = jnp.asarray(rng.standard_normal((n, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, n))
    state = jnp.stack([pos, jnp.zeros((n, 3))], axis=1)
    config = SimulationConfig(N_particles=n, softening=0.1, softening_kernel="plummer")
    params = SimulationParams(G=1.0)
    acc = direct_acc_matrix(state, mass, config, params, return_potential=False)

    @jax.jit
    def historical(state, mass):
        # direct_acc_matrix before the kernels, verbatim (jitted, like it)
        pos = state[:, 0, :]
        dpos = jax.lax.stop_gradient(pos[:, None, :] - pos[None, :, :])
        eye = jax.lax.stop_gradient(jnp.eye(n))
        r2_safe = jnp.sum(dpos**2, axis=-1) + config.softening**2
        inv_r3 = r2_safe**-1.5 * (1.0 - eye)
        return -params.G * jnp.sum((mass[:, None] * dpos) * inv_r3[:, :, None], axis=1)

    assert np.array_equal(np.asarray(acc), np.asarray(historical(state, mass)))
