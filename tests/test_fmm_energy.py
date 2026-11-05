"""``odisseo.utils.E_pot`` / ``E_tot`` on ``FMM_ACC``.

Until 2026-11 they had no ``FMM_ACC`` branch (the FMM notebook's energy cells
crashed). The branch evaluates jaccpot's FMM potential with the solver settings of
the lane the config resolves to (``runner_coupling.runner_self_gravity``: the runner
lane's settings, the fused lane's O(N) potential). Pinned here:

* against the direct-sum ``E_pot`` at small N, per particle and summed, within the
  FMM's tolerance (CPU, N = 512, the runner's small-N settings: leaf 16, p4, theta
  0.6, the geometric MAC), with an external potential;
* the convention is the direct schemes': ``m_i phi_i`` per particle (each pair
  counted twice in the sum) plus ``m_i Phi_ext``;
* under ``jax.jit`` and ``jax.vmap`` (a ``pure_callback``, one evaluation per mapped
  state) its FMM part is the eager call's, bit for bit;
* ``E_tot`` is ``E_kin + E_pot``; the direct schemes are unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from odisseo.option_classes import (
    DIRECT_ACC,
    DIRECT_ACC_MATRIX,
    FMM_ACC,
    NFW_POTENTIAL,
    NFWParams,
    SimulationConfig,
    SimulationParams,
)
from odisseo.utils import E_kin, E_pot, E_tot

pytest.importorskip("jaccpot.production")

N = 512
SOFTENING = 1e-2


def _system():
    """Four blobs of total mass 1 (``test_runner_lane``'s draw), float64."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    masses = masses / masses.sum()
    return (
        jnp.asarray(np.stack([pos, vel], axis=1), jnp.float64),
        jnp.asarray(masses, jnp.float64),
    )


def _config(scheme=FMM_ACC, **overrides):
    values = dict(
        N_particles=N,
        acceleration_scheme=scheme,
        softening=SOFTENING,
        external_accelerations=(NFW_POTENTIAL,),
    )
    values.update(overrides)
    return SimulationConfig(**values)


PARAMS = SimulationParams(G=1.0, NFW_params=NFWParams(Mvir=10.0, r_s=1.0))


@pytest.fixture(scope="module")
def system():
    return _system()


@pytest.fixture(scope="module")
def fmm_energy(system):
    state, masses = system
    return np.asarray(E_pot(state, masses, _config(), PARAMS))


def test_fmm_e_pot_follows_the_direct_sum(system, fmm_energy):
    state, masses = system
    direct = np.asarray(E_pot(state, masses, _config(DIRECT_ACC_MATRIX), PARAMS))
    assert fmm_energy.shape == (N,) and fmm_energy.dtype == np.float64
    rel_l2 = np.linalg.norm(fmm_energy - direct) / np.linalg.norm(direct)
    rel_sum = abs(fmm_energy.sum() - direct.sum()) / abs(direct.sum())
    # leaf 16, p4, theta 0.6 at N = 512 (measured 2026-11-06: see the report)
    assert rel_l2 < 1e-3, rel_l2
    assert rel_sum < 1e-3, rel_sum


def test_the_convention_is_the_direct_schemes(system, fmm_energy):
    """Self part ``m_i phi_i`` (the pair counted twice in the sum), external part
    ``m_i Phi_ext``: without the external field the FMM part alone is left."""
    state, masses = system
    no_field = np.asarray(
        E_pot(state, masses, _config(external_accelerations=()), PARAMS)
    )
    external = fmm_energy - no_field
    direct_no_field = np.asarray(
        E_pot(state, masses, _config(DIRECT_ACC, external_accelerations=()), PARAMS)
    )
    direct_external = (
        np.asarray(E_pot(state, masses, _config(DIRECT_ACC), PARAMS))
        - direct_no_field
    )
    np.testing.assert_allclose(external, direct_external, rtol=1e-12, atol=1e-15)
    assert np.all(no_field < 0)


def test_under_jit_and_vmap_it_is_the_eager_value(system, fmm_energy):
    """The FMM part (the callback) bit for bit; with the external field, which XLA
    fuses differently inside a jit, to round-off."""
    state, masses = system
    cfg = _config(external_accelerations=())
    eager = np.asarray(E_pot(state, masses, cfg, PARAMS))
    jitted = jax.jit(lambda s, m: E_pot(s, m, cfg, PARAMS))(state, masses)
    assert np.array_equal(np.asarray(jitted), eager)
    shifted = state.at[:, 0, 0].add(0.01)
    batch = jnp.stack([state, shifted])
    mapped = np.asarray(jax.vmap(lambda s: E_pot(s, masses, cfg, PARAMS))(batch))
    assert np.array_equal(mapped[0], eager)
    assert np.array_equal(mapped[1], np.asarray(E_pot(shifted, masses, cfg, PARAMS)))
    field = _config()
    with_field = jax.jit(lambda s, m: E_pot(s, m, field, PARAMS))(state, masses)
    np.testing.assert_allclose(np.asarray(with_field), fmm_energy, rtol=1e-13, atol=0)
    # the notebook's form: the summed energy per snapshot
    sums = jax.vmap(lambda s: jnp.sum(E_tot(s, masses, field, PARAMS)))(batch)
    assert np.isfinite(np.asarray(sums)).all()


def test_e_tot_is_e_kin_plus_e_pot(system, fmm_energy):
    state, masses = system
    total = np.asarray(E_tot(state, masses, _config(), PARAMS))
    np.testing.assert_array_equal(
        total, np.asarray(E_kin(state, masses)) + fmm_energy
    )
    direct = _config(DIRECT_ACC_MATRIX)
    np.testing.assert_allclose(
        np.asarray(E_tot(state, masses, direct, PARAMS)),
        np.asarray(E_kin(state, masses)) + np.asarray(E_pot(state, masses, direct, PARAMS)),
        rtol=1e-15,
        atol=0,
    )
