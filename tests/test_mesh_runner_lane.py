"""The ``fmm_mesh_runner`` lane: ``integrate()`` on jaccpot's runner across devices.

``fmm_mesh=True`` with ``fmm_preset="galaxy"`` (or a ``MeshRunnerOptions`` as
``mesh_options``) runs jaccpot's ``ProductionRunner`` in its multi-GPU mode
(:mod:`odisseo.mesh_runner_coupling`); a ``MeshOptions`` keeps the older mesh lane
(:mod:`odisseo.mesh_coupling`, ``tests/test_mesh_lane.py``). Pinned here:

* dispatch: which configs reach the lane and which keep the older one, through
  ``resolve_lane`` and ``integrate(..., return_lane=True)`` -- a ``MeshOptions``
  still reaches ``integrate_mesh_jaccpot`` with its options, whatever the preset;
* what the mesh mode does not run is refused, naming it;
* THIN CLIENT: ``integrate()`` computes what ``ProductionRunner(mesh=...)`` called
  directly computes, bit for bit (an NFW halo, its parameters traced), and returns
  the final state as a host array in input order;
* a second call with another ``dt`` and halo builds no solver and adds no program to
  jaccpot's program cache;
* history runs: the states every ``num_timesteps // num_snapshots`` steps from the
  initial one, bitwise the plain runs of those lengths, and the energy and angular
  momentum that came through the edges are the host fp64 sums over the gathered
  states;
* with a direct-sum force the lane follows Odisseo's fp64 direct lane (same
  velocity Verlet, halo, G, softening and kernel).

The fused multi-GPU force prepares on a GPU only and compiles for minutes on CPU
devices (jaccpot gates it through the runner in a slow job of its own,
``tests/distributed/test_production_runner_mesh_fused.py``), so the integrating tests
inject jaccpot's ``DirectMeshForce``: the direct sum over every device's particles in
global-id order, whose answer does not depend on the partition. What this lane adds
is the plumbing around the runner; ``docs/source/parallelism.md`` records the fused
force through it on two GPUs. N = 512 in four blobs, float64, two FORCED CPU devices;
the device count is fixed when the backend starts, so this file runs on its own::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \\
        python -m pytest tests/test_mesh_runner_lane.py -n 0
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from odisseo import mesh_coupling
from odisseo import mesh_runner_coupling as mrc
from odisseo.integration_api import INTEGRATION_LANES, integrate, resolve_lane
from odisseo.mesh_coupling import MeshOptions
from odisseo.mesh_runner_coupling import MeshRunnerOptions
from odisseo.option_classes import (
    DIRECT_ACC,
    FMM_ACC,
    NFW_POTENTIAL,
    NFWParams,
    SimulationConfig,
    SimulationParams,
)
from odisseo.softening import softened_inverse_powers

jaccpot_production = pytest.importorskip("jaccpot.production")
GalaxySettings = jaccpot_production.GalaxySettings

needs_two = pytest.mark.skipif(
    jax.device_count() < 2,
    reason="needs 2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)",
)

N = 512
NDEV = 2
SOFTENING = 1e-2
#: a power of two, so t_end / num_timesteps is exact for every run length below
DT = 2.0**-10
#: the single-card runner tests' small-N operating point; the injected direct force
#: ignores it (it is what the solver would be built with)
SMALL_N = GalaxySettings(
    leaf_size=16,
    max_order=4,
    theta=0.6,
    cell_min_level=None,
    mac_type="dehnen",
    adaptive_eps=None,
)


def _direct_force(G=1.0):
    """jaccpot's partition-independent direct-sum mesh force, the lane's kernel."""
    from jaccpot.distributed.segment import DirectMeshForce

    return DirectMeshForce(G=G, softening=SOFTENING, softening_kernel="ferrers3")


#: one force object for the module: it is the cache owner of its programs, and the
#: options holding it are part of the kept runner's key
FORCE = None


def _force():
    global FORCE
    if FORCE is None:
        FORCE = _direct_force()
    return FORCE


def _system():
    """Four blobs of total mass 1 (the single-card runner tests' draw), float64."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    masses = masses / masses.sum()
    return np.stack([pos, vel], axis=1), masses


def _config(**overrides):
    values = dict(
        N_particles=N,
        acceleration_scheme=FMM_ACC,
        num_timesteps=6,
        softening=SOFTENING,
        fmm_preset="galaxy",
        fmm_use_pallas=False,
        fmm_mesh=True,
        external_accelerations=(NFW_POTENTIAL,),
        fmm_galaxy_settings=SMALL_N,
    )
    if "mesh_options" not in overrides:
        values["mesh_options"] = MeshRunnerOptions(ndev=NDEV, force=_force())
    values.update(overrides)
    return SimulationConfig(**values)


def _params(steps=6, *, G=1.0, mvir=10.0):
    return SimulationParams(
        G=G, t_end=steps * DT, NFW_params=NFWParams(Mvir=mvir, r_s=1.0)
    )


def _bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, (what, got.shape, want.shape)
    differ = int(np.sum(got != want))
    assert differ == 0, f"{what}: {differ} of {got.size} elements differ"


# --------------------------------------------------------------------------
# Dispatch and options: no integration.
# --------------------------------------------------------------------------

_OLD = MeshOptions(dt=1e-3)


@pytest.mark.parametrize(
    "overrides, lane",
    [
        # the production settings ask for the runner's multi-GPU mode
        (dict(fmm_preset="galaxy"), "fmm_mesh_runner"),
        (
            dict(fmm_preset="galaxy", mesh_options=MeshRunnerOptions()),
            "fmm_mesh_runner",
        ),
        (dict(mesh_options=MeshRunnerOptions(ndev=2)), "fmm_mesh_runner"),
        # what the mesh mode does not run still reaches it, and is refused there
        (dict(fmm_preset="galaxy", fmm_differentiable=True), "fmm_mesh_runner"),
        (dict(fmm_preset="galaxy", fmm_blockstep=True), "fmm_mesh_runner"),
        # a MeshOptions keeps the older lane, whatever the preset
        (dict(mesh_options=_OLD), "fmm_mesh"),
        (dict(fmm_preset="galaxy", mesh_options=_OLD), "fmm_mesh"),
        (dict(fmm_preset="large_n_gpu", mesh_options=_OLD), "fmm_mesh"),
        (dict(), "fmm_mesh"),
        # and without fmm_mesh the single-card lanes
        (dict(fmm_mesh=False, fmm_preset="galaxy"), "fmm_runner"),
        (dict(fmm_mesh=False, mesh_options=MeshRunnerOptions()), "fmm_forward"),
    ],
)
def test_resolve_lane_names_the_mesh_runner_lane(overrides, lane):
    values = {"acceleration_scheme": FMM_ACC, "fmm_mesh": True, **overrides}
    cfg = SimulationConfig(**values)
    assert resolve_lane(cfg, backend="cpu", num_particles=10) == lane
    assert lane in INTEGRATION_LANES


def test_integrate_dispatches_on_the_lane_it_reports(monkeypatch):
    """``fmm_mesh`` with the galaxy preset reaches the runner's mesh entry point, a
    ``MeshOptions`` the older lane's ``integrate_mesh_jaccpot`` with its options and
    step count, as before; ``integrate`` reports the lane that ran."""
    calls = []

    def fake_runner(state, mass, config, params, *, scheduled):
        calls.append(("runner", scheduled))
        return mrc.RunnerResult(state=state, snapshots=None, run=None)

    def fake_old(state, mass, config, params, *, options, n_steps):
        calls.append(("old", options, n_steps))
        return mesh_coupling.MeshResult(state=state)

    monkeypatch.setattr(mrc, "integrate_mesh_runner_jaccpot", fake_runner)
    monkeypatch.setattr(mesh_coupling, "integrate_mesh_jaccpot", fake_old)
    state = jnp.zeros((6, 2, 3), jnp.float32)
    mass = jnp.ones((6,), jnp.float32)
    cfg = SimulationConfig(
        N_particles=6,
        acceleration_scheme=FMM_ACC,
        num_timesteps=3,
        fmm_mesh=True,
        fmm_preset="galaxy",
    )
    params = SimulationParams()
    lane = integrate(state, mass, cfg, params, return_lane=True)[1]
    assert lane == "fmm_mesh_runner"
    lane = integrate(
        state, mass, cfg._replace(mesh_options=_OLD), params, return_lane=True
    )[1]
    assert lane == "fmm_mesh"
    assert calls == [("runner", False), ("old", _OLD, 3)]


@pytest.mark.parametrize(
    "kw", [dict(ndev=0), dict(repartition_every=-1), dict(cap_headroom=0.9)]
)
def test_mesh_runner_options_validate(kw):
    with pytest.raises(ValueError):
        MeshRunnerOptions(**kw)


def test_the_options_are_hashable_and_default_to_every_device():
    assert hash(_config()) == hash(_config())
    assert mrc.mesh_runner_options(_config(mesh_options=None)) == MeshRunnerOptions()
    with pytest.raises(TypeError, match="MeshRunnerOptions"):
        mrc.mesh_runner_options(_config(mesh_options={"ndev": 2}))


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(fixed_timestep=False), "fixed_timestep"),
        (dict(fmm_refresh_every=2), "fmm_refresh_every"),
        (dict(fmm_refresh_after_position_update=True), "after_position"),
        (dict(fmm_differentiable=True), "gradient"),
        (dict(fmm_blockstep=True), "block steps"),
    ],
)
def test_what_the_mesh_mode_does_not_run_is_refused(overrides, match):
    state, masses = _system()
    with pytest.raises(NotImplementedError, match=match):
        integrate(state, masses, _config(**overrides), _params())


def test_schedules_step_callbacks_and_gradients_are_refused():
    state, masses = _system()
    with pytest.raises(NotImplementedError, match="active-particle"):
        integrate(
            state,
            masses,
            _config(),
            _params(),
            active_indices_schedule=jnp.zeros((6, 4), jnp.int32),
        )
    with pytest.raises(NotImplementedError, match="step_callback"):
        mrc.integrate_mesh_runner_jaccpot(
            state, masses, _config(), _params(), step_callback=lambda *a: None
        )

    def loss(mvir):
        out = integrate(
            jnp.asarray(state), jnp.asarray(masses), _config(), _params(mvir=mvir)
        )
        return jnp.sum(out)

    with pytest.raises(NotImplementedError, match="traced"):
        jax.grad(loss)(10.0)


def test_the_fused_force_needs_a_gpu_and_the_devices_must_exist():
    state, masses = _system()
    with pytest.raises(NotImplementedError, match="GPU backend"):
        integrate(state, masses, _config(mesh_options=None), _params())
    many = MeshRunnerOptions(ndev=jax.device_count() + 1, force=_force())
    with pytest.raises(ValueError, match="device"):
        integrate(state, masses, _config(mesh_options=many), _params())


@needs_two
def test_another_g_than_the_forces_is_refused():
    """The mesh force is static in G and the softening; the lane passes them as
    force parameters, so the runner refuses an injected force with others."""
    state, masses = _system()
    with pytest.raises(ValueError, match="static in G"):
        integrate(state, masses, _config(), _params(G=1.1))
    mrc.clear_mesh_runner_cache()


# --------------------------------------------------------------------------
# Integrating tests (two CPU devices, the direct-sum mesh force).
# --------------------------------------------------------------------------


def _jaccpot_mesh_runner_reference(state, masses, steps, params):
    """``ProductionRunner(mesh=...)`` called directly: a solver from jaccpot's API,
    the halo as a closure over Odisseo's potential, its parameters as float64
    arrays, its own direct-sum force."""
    from jaccpot import FastMultipoleMethod, ProductionRunner, SegmentConfig
    from jaccpot.production import MeshRunConfig
    from yggdrax.distributed import make_mesh

    from odisseo.potentials import combined_external_acceleration_vmpa_switch

    halo_config = SimulationConfig(external_accelerations=(NFW_POTENTIAL,))

    def halo(state_in, p):
        return combined_external_acceleration_vmpa_switch(state_in, halo_config, p)

    runner = ProductionRunner(
        FastMultipoleMethod(preset="galaxy", G=1.0, softening=SOFTENING),
        settings=SMALL_N,
        segment=SegmentConfig(steps=50, diagnostics_every=0),
        external_acceleration_fn=halo,
        mesh=MeshRunConfig(make_mesh(NDEV), force=_direct_force()),
    )
    p64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), params)
    return runner.run(state, masses, dt=DT, num_steps=steps, external_params=p64)


def _programs(force):
    from jaccpot.distributed.segment import _PROGRAMS_ATTR

    return set(getattr(force.cache_owner, _PROGRAMS_ATTR, {}))


@needs_two
def test_integrate_is_the_jaccpot_mesh_runner_and_a_second_call_compiles_nothing():
    state, masses = _system()
    cfg = _config()

    out, lane = integrate(state, masses, cfg, _params(), return_lane=True)
    assert lane == "fmm_mesh_runner"
    # the final state on the host, in input order
    assert isinstance(out, np.ndarray) and out.shape == (N, 2, 3)
    ref = _jaccpot_mesh_runner_reference(state, masses, 6, _params())
    assert ref.mode == "mesh"
    _bitwise(out, ref.state, "final state")

    # another dt and halo: the kept solver and runner, no new program
    ((solver, runner),) = mrc._CACHE.values()
    programs = _programs(_force())
    assert programs
    second = integrate(state, masses, cfg, _params(steps=12, mvir=12.0))
    ((solver_2, runner_2),) = mrc._CACHE.values()
    assert solver_2 is solver and runner_2 is runner
    assert _programs(_force()) == programs
    assert not np.array_equal(second, out)  # the values took effect
    # and the first values again: the first call's state
    _bitwise(integrate(state, masses, cfg, _params()), out, "repeated call")

    # another G is another solver: the mesh force is static in it (looked up only)
    settings = mrc.galaxy_settings(cfg, np.float64)
    options = mrc.mesh_runner_options(cfg)
    solver_3, _ = mrc._solver_and_runner(
        cfg, _params(G=1.1), settings, options, NDEV, N, True
    )
    assert solver_3 is not solver
    mrc.clear_mesh_runner_cache()


@needs_two
def test_history_runs_return_every_kth_state_with_the_diagnostics():
    """6 steps, 3 snapshots: the states at steps 0, 2 and 4 (2-step segments), at
    times 0, 2 dt, 4 dt, bitwise the plain runs of 2 and 4 steps in segments of 2;
    the energy and angular momentum that came through the edges are the host fp64
    energy and angular momentum of the gathered states."""
    from jaccpot import SegmentConfig

    from odisseo.potentials import combined_external_acceleration_vmpa_switch

    state, masses = _system()
    cfg = _config(return_snapshots=True, num_snapshots=3)
    snaps, lane = integrate(state, masses, cfg, _params(), return_lane=True)
    assert lane == "fmm_mesh_runner"
    # the kept runner keeps no snapshot (nor the collector) for the next call
    ((_, runner),) = mrc._CACHE.values()
    assert runner.callbacks == ()
    assert isinstance(snaps.states, np.ndarray) and snaps.states.shape == (3, N, 2, 3)
    np.testing.assert_array_equal(snaps.times, [0.0, 2 * DT, 4 * DT])
    _bitwise(snaps.states[0], state, "snapshot 0")
    for k, steps in ((1, 2), (2, 4)):
        plain = mrc.integrate_mesh_runner_jaccpot(
            state,
            masses,
            _config(num_timesteps=steps),
            _params(steps=steps),
            segment=SegmentConfig(steps=2, diagnostics_every=0),
        )
        _bitwise(snaps.states[k], plain.state, f"snapshot {k}")
    assert snaps.num_iterations == 6

    # the host fp64 sums over each gathered state
    field = SimulationConfig(external_accelerations=(NFW_POTENTIAL,))
    for k, snap in enumerate(snaps.states):
        x, v = snap[:, 0], snap[:, 1]
        kinetic = 0.5 * np.sum(masses * np.sum(v * v, axis=1))
        d = x[:, None, :] - x[None, :, :]
        _, psi = softened_inverse_powers(
            np.sum(d * d, axis=-1), SOFTENING, "ferrers3", xp=np
        )
        np.fill_diagonal(psi, 0.0)
        w_self = -0.5 * np.sum(masses[:, None] * masses[None, :] * psi)
        _, phi_ext = combined_external_acceleration_vmpa_switch(
            jnp.asarray(snap), field, _params(), return_potential=True
        )
        w_ext = np.sum(masses * np.asarray(phi_ext))
        np.testing.assert_allclose(
            snaps.total_energy[k], kinetic + w_self + w_ext, rtol=1e-12
        )
        expected = np.sum(masses[:, None] * np.cross(x, v), axis=0)
        np.testing.assert_allclose(
            snaps.angular_momentum[k], expected, rtol=1e-12, atol=1e-15
        )
    mrc.clear_mesh_runner_cache()


@needs_two
def test_the_mesh_runner_lane_follows_the_direct_lane():
    """With the direct-sum force, 6 steps with the halo stay with Odisseo's fp64
    direct lane (the same velocity Verlet, kernel, G and softening) to round-off
    (measured: velocity change 4.3e-16, positions equal): dt, the field and its
    parameters, and the state's order reach the mesh as on one card."""
    state, masses = _system()
    params = _params()
    direct = np.asarray(
        integrate(
            jnp.asarray(state),
            jnp.asarray(masses),
            _config(acceleration_scheme=DIRECT_ACC),
            params,
        )
    )
    mesh = integrate(state, masses, _config(), params)
    change = np.linalg.norm(direct[:, 1] - state[:, 1])
    dv = np.linalg.norm(mesh[:, 1] - direct[:, 1]) / change
    dx = np.linalg.norm(mesh[:, 0] - direct[:, 0]) / np.linalg.norm(direct[:, 0])
    assert dv < 1e-12, dv
    assert dx < 1e-14, dx
    mrc.clear_mesh_runner_cache()


@needs_two
def test_repartition_cadence_and_headroom_reach_the_runner():
    """``MeshRunnerOptions`` is the mesh layout of jaccpot's ``MeshRunConfig``: the
    shard capacity from the headroom, the repartitions at the cadence; the result is
    the same run (the direct force is partition-independent)."""
    from jaccpot import SegmentConfig

    state, masses = _system()
    segment = SegmentConfig(steps=2, diagnostics_every=0)
    every = mrc.integrate_mesh_runner_jaccpot(
        state, masses, _config(), _params(), segment=segment
    )
    never = mrc.integrate_mesh_runner_jaccpot(
        state,
        masses,
        _config(
            mesh_options=MeshRunnerOptions(
                ndev=NDEV, repartition_every=0, cap_headroom=1.5, force=_force()
            )
        ),
        _params(),
        segment=segment,
    )
    cap = jaccpot_production.mesh_shard_capacity(N, NDEV)
    assert [e.plan["mesh"]["cap"] for e in every.run.edges] == [cap] * 4
    assert [e.plan["mesh"]["cap"] for e in never.run.edges] == [
        jaccpot_production.mesh_shard_capacity(N, NDEV, 1.5)
    ] * 4
    moved = [e.plan["repartition"] for e in every.run.edges[1:]]
    assert [r["repartitioned"] for r in moved] == [False, True, True]
    assert not any(e.plan["repartition"]["repartitioned"] for e in never.run.edges[1:])
    np.testing.assert_allclose(every.state, never.state, rtol=0, atol=1e-14)
    mrc.clear_mesh_runner_cache()
