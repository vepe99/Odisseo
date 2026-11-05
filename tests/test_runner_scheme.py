"""``config.fmm_scheme``: nornax's integration scheme in jaccpot's runner (plan I1).

The multi-node accuracy plan's U1: nornax owns the integration schemes, jaccpot's
``ProductionRunner`` the run, Odisseo the science. Pinned here:

* routing: a fixed-step ``FMM_ACC`` run is the runner's (since the 2026-11 switch,
  with or without a scheme); the lanes with their own step (the adaptive one
  included) refuse a scheme; the multi-GPU runner lane refuses it until I2;
* ``runner_scheme``: ``None`` is the runner's built-in KDK, ``"nornax_kdk"``
  nornax's ``LeapfrogKDK``, other names and non-schemes are refused;
* THE GATE: ``fmm_scheme="nornax_kdk"`` computes what the built-in runner lane
  computes, bit for bit (an NFW halo, its parameters traced; 6 steps), through
  jaccpot programs of its own;
* the deprecations: Odisseo's own FMM leapfrog loop and the older mesh lane's KDK
  warn ``DeprecationWarning`` once, on the caller's line, naming the replacement.

Small-N operating point and CPU trick as ``test_runner_lane.py``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from odisseo import integration_api as api
from odisseo import runner_coupling as rc
from odisseo.integration_api import integrate, resolve_lane
from odisseo.mesh_coupling import MESH_KDK_DEPRECATION, MeshOptions
from odisseo.mesh_runner_coupling import mesh_runner_unsupported
from odisseo.option_classes import FMM_ACC, SimulationConfig, SimulationParams

pytest.importorskip("jaccpot.scheme")
nornax_schemes = pytest.importorskip("nornax.schemes")

from test_runner_lane import (  # noqa: E402
    _config,
    _params,
    _strict_programs,
    _system,
    fused_lane,  # noqa: F401
)


@pytest.mark.parametrize(
    "overrides, kw, lane",
    [
        # a scheme asks for the runner, as "galaxy" does, on any backend
        (
            dict(fmm_scheme="nornax_kdk"),
            dict(backend="cpu", num_particles=10),
            "fmm_runner",
        ),
        (
            dict(fmm_scheme="nornax_kdk", fmm_preset="large_n_gpu"),
            dict(backend="gpu", num_particles=10),
            "fmm_runner",
        ),
        # without a scheme too, since the 2026-11 switch (the built-in KDK)
        (dict(), dict(backend="cpu", num_particles=10), "fmm_runner"),
        # the multi-GPU runner lane is selected (and refuses the scheme there)
        (
            dict(fmm_scheme="nornax_kdk", fmm_mesh=True, fmm_preset="galaxy"),
            dict(),
            "fmm_mesh_runner",
        ),
    ],
)
def test_a_scheme_selects_the_runner_lane(overrides, kw, lane):
    cfg = SimulationConfig(**{"acceleration_scheme": FMM_ACC, **overrides})
    assert resolve_lane(cfg, **kw) == lane


@pytest.mark.parametrize(
    "overrides",
    [
        dict(fmm_blockstep=True),
        dict(fmm_differentiable=True),
        dict(fmm_mesh=True, mesh_options=MeshOptions(dt=1e-3)),
        dict(fixed_timestep=False),
    ],
    ids=["blockstep", "differentiable", "older_mesh", "adaptive"],
)
def test_the_lanes_with_their_own_step_refuse_a_scheme(overrides):
    cfg = SimulationConfig(
        acceleration_scheme=FMM_ACC, fmm_scheme="nornax_kdk", **overrides
    )
    with pytest.raises(ValueError, match="has its own step"):
        resolve_lane(cfg)


def test_the_multi_gpu_runner_refuses_a_scheme_until_i2():
    cfg = SimulationConfig(
        acceleration_scheme=FMM_ACC, fmm_preset="galaxy", fmm_mesh=True
    )
    assert mesh_runner_unsupported(cfg) is None
    reason = mesh_runner_unsupported(cfg._replace(fmm_scheme="nornax_kdk"))
    assert reason is not None and "I2" in reason


def test_runner_scheme_names():
    assert rc.runner_scheme(SimulationConfig()) is None
    assert rc.runner_scheme(SimulationConfig(fmm_scheme="nornax_kdk")) == (
        nornax_schemes.LeapfrogKDK()
    )
    given = nornax_schemes.LeapfrogKDK()
    assert rc.runner_scheme(SimulationConfig(fmm_scheme=given)) is given
    with pytest.raises(ValueError, match="nornax_kdk"):
        rc.runner_scheme(SimulationConfig(fmm_scheme="hermite"))
    with pytest.raises(TypeError, match="SplitStepScheme"):
        rc.runner_scheme(SimulationConfig(fmm_scheme=(1, 2)))


def test_the_nornax_kdk_is_the_builtin_runner_lane(fused_lane):  # noqa: F811
    """The gate: nornax's KDK in the runner IS the runner's built-in KDK, bit for
    bit, through programs of its own (the scheme is in the compile key)."""
    state_np, masses_np = _system()
    state = jnp.asarray(state_np, jnp.float32)
    masses = jnp.asarray(masses_np, jnp.float32)
    rc.clear_runner_cache()
    builtin, lane = integrate(state, masses, _config(), _params(), return_lane=True)
    assert lane == "fmm_runner"
    out, lane = integrate(
        state, masses, _config(fmm_scheme="nornax_kdk"), _params(), return_lane=True
    )
    assert lane == "fmm_runner"
    assert np.array_equal(np.asarray(out), np.asarray(builtin))
    ((solver, runner),) = rc._CACHE.values()
    assert runner.scheme == nornax_schemes.LeapfrogKDK()
    programs = _strict_programs(solver)
    assert programs and all(
        ("scheme", nornax_schemes.LeapfrogKDK().key()) in k for k in programs
    )
    rc.clear_runner_cache()


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_a_default_fixed_step_run_warns_no_deprecation(monkeypatch):
    """Since the 2026-11 switch ``integrate()`` no longer reaches the deprecated
    loop: a fixed-step run is the runner's, without a ``DeprecationWarning``."""

    def fake_runner(state, mass, config, params, *, scheduled):
        return rc.RunnerResult(state=state, snapshots=None, run=None)

    monkeypatch.setattr(rc, "integrate_runner_jaccpot", fake_runner)
    state = jnp.zeros((6, 2, 3), jnp.float32)
    cfg = SimulationConfig(N_particles=6, acceleration_scheme=FMM_ACC, num_timesteps=2)
    (_, lane), caught = _caught(
        lambda: integrate(state, jnp.ones(6), cfg, SimulationParams(), return_lane=True)
    )
    assert lane == "fmm_runner"
    assert caught == []


def test_the_loop_warns_when_called_directly():
    from odisseo.jaccpot_coupling import integrate_leapfrog_jaccpot_active

    cfg = SimulationConfig(N_particles=6, acceleration_scheme=FMM_ACC)
    with pytest.warns(DeprecationWarning, match="fmm_runner"):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            integrate_leapfrog_jaccpot_active(
                jnp.zeros((6, 2, 3)), jnp.ones(6), cfg, SimulationParams(), num_steps=0
            )


def test_the_older_mesh_lane_warns_once_on_the_callers_line(monkeypatch):
    def fake_old():
        # the lane's own warning, as integrate_mesh_jaccpot issues it
        warnings.warn(MESH_KDK_DEPRECATION, DeprecationWarning, stacklevel=2)

    calls = []

    def fake_lane(state, mass, config, params):
        calls.append(config.mesh_options)
        fake_old()
        return state

    monkeypatch.setattr(api, "_integrate_fmm_mesh", fake_lane)
    state = jnp.zeros((6, 2, 3), jnp.float32)
    cfg = SimulationConfig(
        N_particles=6,
        acceleration_scheme=FMM_ACC,
        fmm_mesh=True,
        mesh_options=MeshOptions(dt=1e-3),
    )
    (_, lane), caught = _caught(
        lambda: integrate(state, jnp.ones(6), cfg, SimulationParams(), return_lane=True)
    )
    assert lane == "fmm_mesh" and len(calls) == 1
    assert len(caught) == 1 and caught[0].filename == __file__
    assert "fmm_mesh_runner" in str(caught[0].message)
