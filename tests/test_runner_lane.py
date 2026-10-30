"""The ``fmm_runner`` lane: ``integrate()`` as a thin client of jaccpot's runner.

``fmm_preset="galaxy"`` -- and the automatic large-N profile, which picked the
coupler's ``"large_n_gpu"`` loop before -- runs jaccpot's ``ProductionRunner``
(:mod:`odisseo.runner_coupling`). Pinned here:

* dispatch: which configs reach the lane, and which keep the coupler (``resolve_lane``
  is the single source of truth, so it is checked directly and through
  ``integrate(..., return_lane=True)``);
* the settings: ``GalaxySettings`` by default, ``fmm_galaxy_settings`` when given,
  ``fmm_mac_type="dehnen_error"`` honoured, other MACs refused; the solver built
  from ``GALAXY_SETTINGS`` is ``preset="galaxy"``'s;
* without a MAC of its own the lane runs the preset's -- Dehnen's eq (16a) at 1e-4
  since jaccpot's 2026-10-21 preset change, the geometric MAC before -- bitwise
  jaccpot's runner on ``preset="galaxy"``, and under ``dehnen_error`` another G
  builds another solver;
* options the runner does not run are refused on an explicit ``"galaxy"``;
* THIN CLIENT: ``integrate()`` computes what ``ProductionRunner`` called directly
  computes, bit for bit (an NFW halo, its parameters traced);
* a second call with another ``dt``, ``G`` and halo builds no solver and adds no
  program to jaccpot's compile cache (on the geometric MAC, pinned: under
  ``dehnen_error`` G is the solver's);
* history runs: the state every ``num_timesteps // num_snapshots`` steps from the
  initial one, bitwise the plain runs of those lengths, with the energy and the
  angular momentum;
* (slow) the physics: at a tight opening angle the lane follows Odisseo's fp64
  direct-sum lane, as the coupler's CPU lane does.

The fused lane's prepare runs on a GPU backend only, so the integrating tests report
``jax.default_backend()`` as ``"gpu"`` -- jaccpot's own CPU tests of the lane do the
same (its pure-JAX near field, no Pallas). N = 512 in four blobs, total mass 1, at a
small-N operating point (leaf 16, p4, theta 0.6, no cell level floor), like jaccpot's
runner tests, on the geometric MAC unless a test says otherwise.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from odisseo import integration_api as api
from odisseo import runner_coupling as rc
from odisseo.integration_api import integrate, resolve_lane
from odisseo.mesh_coupling import MeshOptions
from odisseo.option_classes import (
    DIRECT_ACC,
    FMM_ACC,
    NFW_POTENTIAL,
    NFWParams,
    SimulationConfig,
    SimulationParams,
)

jaccpot_production = pytest.importorskip("jaccpot.production")
GalaxySettings = jaccpot_production.GalaxySettings
GALAXY_SETTINGS = jaccpot_production.GALAXY_SETTINGS

N = 512
SOFTENING = 1e-2
#: a power of two, so t_end / num_timesteps is exact for every run length below
DT = 2.0**-10
#: the small-N operating point on the geometric MAC, explicitly: jaccpot's preset MAC
#: is ``dehnen_error`` since 2026-10-21, which takes the solver's G only (a new G
#: compiles) and needs the CSR near field
SMALL_N = GalaxySettings(
    leaf_size=16,
    max_order=4,
    theta=0.6,
    cell_min_level=None,
    mac_type="dehnen",
    adaptive_eps=None,
)
#: the same operating point with the preset's own MAC
SMALL_N_PRESET_MAC = dataclasses.replace(
    GALAXY_SETTINGS, leaf_size=16, max_order=4, theta=0.6, cell_min_level=None
)
#: the CSR near field in Pallas interpret mode: dehnen_error's force scale needs it
#: (jaccpot's runner tests switch the same)
_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


def _system():
    """Four blobs of total mass 1 (jaccpot's runner-test draw), float32."""
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
        external_accelerations=(NFW_POTENTIAL,),
        fmm_galaxy_settings=SMALL_N,
    )
    values.update(overrides)
    return SimulationConfig(**values)


def _params(steps=6, *, G=1.0, mvir=10.0):
    return SimulationParams(
        G=G, t_end=steps * DT, NFW_params=NFWParams(Mvir=mvir, r_s=1.0)
    )


@pytest.fixture
def fused_lane(monkeypatch):
    """Report the GPU backend, so the fused lane's prepare runs on CPU."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")


# --------------------------------------------------------------------------
# Dispatch and settings: no integration.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, kw, lane",
    [
        (
            dict(fmm_preset="galaxy"),
            dict(backend="cpu", num_particles=10),
            "fmm_runner",
        ),
        (dict(), dict(backend="gpu", num_particles=200_000), "fmm_runner"),
        (dict(), dict(backend="gpu", num_particles=199_999), "fmm_forward"),
        (dict(), dict(backend="cpu", num_particles=10**7), "fmm_forward"),
        (dict(N_particles=200_000), dict(backend="gpu"), "fmm_runner"),
        (
            dict(fmm_preset="large_n_gpu"),
            dict(backend="gpu", num_particles=10**7),
            "fmm_forward",
        ),
        (
            dict(fmm_auto_large_n_profile=False),
            dict(backend="gpu", num_particles=10**7),
            "fmm_forward",
        ),
        # the auto profile keeps the coupler for what the runner does not run
        (
            dict(fixed_timestep=False),
            dict(backend="gpu", num_particles=10**7),
            "fmm_forward",
        ),
        (
            dict(fmm_refresh_every=2),
            dict(backend="gpu", num_particles=10**7),
            "fmm_forward",
        ),
        (
            dict(fmm_refresh_after_position_update=True),
            dict(backend="gpu", num_particles=10**7),
            "fmm_forward",
        ),
        (
            dict(),
            dict(backend="gpu", num_particles=10**7, scheduled=True),
            "fmm_forward",
        ),
        # the other FMM lanes keep their precedence
        (
            dict(fmm_preset="galaxy", fmm_differentiable=True),
            dict(),
            "fmm_differentiable",
        ),
        (dict(fmm_preset="galaxy", fmm_blockstep=True), dict(), "fmm_blockstep"),
        # fmm_mesh: the runner's multi-GPU mode, a MeshOptions the older mesh lane
        (dict(fmm_preset="galaxy", fmm_mesh=True), dict(), "fmm_mesh_runner"),
        (
            dict(fmm_preset="galaxy", fmm_mesh=True, mesh_options=MeshOptions(dt=1e-3)),
            dict(),
            "fmm_mesh",
        ),
        (dict(fmm_preset="galaxy", acceleration_scheme=DIRECT_ACC), dict(), "direct"),
    ],
)
def test_resolve_lane_names_the_runner_lane(overrides, kw, lane):
    cfg = SimulationConfig(**{"acceleration_scheme": FMM_ACC, **overrides})
    assert resolve_lane(cfg, **kw) == lane


def test_integrate_dispatches_on_the_lane_it_reports(monkeypatch):
    """The auto profile on a GPU reaches the runner entry point; with an active
    schedule it keeps the coupler's ``large_n_gpu`` loop; ``integrate`` reports the
    lane that ran."""
    calls = []

    def fake_runner(state, mass, config, params, *, scheduled):
        calls.append(("runner", scheduled))
        return rc.RunnerResult(state=state, snapshots=None, run=None)

    def fake_coupler(*args, **kwargs):
        calls.append(("coupler", kwargs["fmm_preset"]))
        return kwargs["state"]

    monkeypatch.setattr(rc, "integrate_runner_jaccpot", fake_runner)
    monkeypatch.setattr(api, "integrate_leapfrog_jaccpot_active", fake_coupler)
    monkeypatch.setattr(api.jax, "default_backend", lambda: "gpu")
    state = jnp.zeros((6, 2, 3), jnp.float32)
    mass = jnp.ones((6,), jnp.float32)
    cfg = SimulationConfig(
        N_particles=6,
        acceleration_scheme=FMM_ACC,
        num_timesteps=3,
        fmm_large_n_min_particles=6,
    )
    params = SimulationParams()
    assert integrate(state, mass, cfg, params, return_lane=True)[1] == "fmm_runner"
    schedule = jnp.zeros((3, 6), jnp.int32)
    out, lane = integrate(
        state, mass, cfg, params, active_indices_schedule=schedule, return_lane=True
    )
    assert lane == "fmm_forward"
    assert calls == [("runner", False), ("coupler", "large_n_gpu")]


def test_the_settings_are_galaxy_settings_unless_given():
    f32, f64 = jnp.float32, jnp.float64
    cfg = SimulationConfig(acceleration_scheme=FMM_ACC, fmm_preset="galaxy")
    # Odisseo's own numerical knobs are not read; its kernel is
    assert rc.galaxy_settings(cfg, f32) == GALAXY_SETTINGS
    assert (
        rc.galaxy_settings(
            cfg._replace(fmm_leaf_size=32, fmm_max_order=8, fmm_theta=0.3), f32
        )
        == GALAXY_SETTINGS
    )
    assert rc.galaxy_settings(cfg._replace(softening_kernel="plummer"), f32) == (
        dataclasses.replace(GALAXY_SETTINGS, softening_kernel="plummer")
    )
    # the coupler's fp32 switch, off with a float64 state: the fp64 force
    assert rc.galaxy_settings(cfg, f64).precision == "fp32"
    assert (
        rc.galaxy_settings(cfg._replace(fmm_large_n_force_fp32=False), f64).precision
        == "fp64"
    )
    assert (
        rc.galaxy_settings(cfg._replace(fmm_large_n_force_fp32=False), f32).precision
        == "fp32"
    )
    # explicit settings win
    assert rc.galaxy_settings(cfg._replace(fmm_galaxy_settings=SMALL_N), f64) == SMALL_N
    # dehnen_error with its target; other MACs refused
    err = rc.galaxy_settings(
        cfg._replace(fmm_mac_type="dehnen_error", fmm_adaptive_eps=1e-4), f32
    )
    assert (err.mac_type, err.adaptive_eps) == ("dehnen_error", 1e-4)
    with pytest.raises(ValueError, match="fmm_adaptive_eps"):
        rc.galaxy_settings(cfg._replace(fmm_mac_type="dehnen_error"), f32)
    with pytest.raises(ValueError, match="Dehnen MAC"):
        rc.galaxy_settings(cfg._replace(fmm_mac_type="bh"), f32)
    with pytest.raises(TypeError, match="GalaxySettings"):
        rc.galaxy_settings(cfg._replace(fmm_galaxy_settings={"theta": 0.5}), f32)


def test_galaxy_settings_build_the_galaxy_preset():
    """``build_galaxy_solver(GALAXY_SETTINGS)`` is ``FastMultipoleMethod(preset=
    "galaxy")``: every engine attribute equal (jaccpot's own preset test compares
    the same way); other settings reach the engine."""
    from jaccpot import FastMultipoleMethod

    def engine(solver):
        return {k: repr(v) for k, v in vars(solver._impl).items()}

    preset = engine(FastMultipoleMethod(preset="galaxy", G=2.0, softening=1e-3))
    built = rc.build_galaxy_solver(GALAXY_SETTINGS, G=2.0, softening=1e-3)
    assert engine(built) == preset
    impl = rc.build_galaxy_solver(SMALL_N, G=1.0, softening=1e-2)._impl
    assert (impl.theta, impl.fixed_order) == (pytest.approx(0.6), 4)
    err = rc.build_galaxy_solver(
        dataclasses.replace(
            GALAXY_SETTINGS, mac_type="dehnen_error", adaptive_eps=3e-5
        ),
        G=1.0,
        softening=1e-3,
    )._impl
    assert err.adaptive_eps == pytest.approx(3e-5)
    assert err._flat_walk_criterion_active()


#: The ``GalaxySettings`` fields ``galaxy_settings`` / ``build_galaxy_solver`` map
#: (``carry`` is the runner's; ``cubic_bounds`` arrives with jaccpot's S3b).
MAPPED_SETTINGS = {
    "leaf_size",
    "max_order",
    "theta",
    "cell_min_level",
    "leaf_capacity",
    "leaf_headroom",
    "cubic_bounds",
    "softening_kernel",
    "mac_type",
    "adaptive_eps",
    "carry",
    "precision",
}


def test_the_builder_knows_every_galaxy_setting():
    """A field jaccpot adds to ``GalaxySettings`` reaches the preset through
    ``_galaxy_advanced``; ``build_galaxy_solver`` keeps the preset's value, so a
    caller's ``fmm_galaxy_settings`` would be ignored for it until it is mapped."""
    fields = {f.name for f in dataclasses.fields(GalaxySettings)}
    assert fields <= MAPPED_SETTINGS, sorted(fields - MAPPED_SETTINGS)


def test_the_cubic_box_setting_reaches_the_engine():
    if not hasattr(GALAXY_SETTINGS, "cubic_bounds"):
        pytest.skip("this jaccpot has no GalaxySettings.cubic_bounds (S3b)")
    for cubic in (True, False):
        settings = dataclasses.replace(GALAXY_SETTINGS, cubic_bounds=cubic)
        solver = rc.build_galaxy_solver(settings, G=1.0, softening=1e-3)
        assert solver.advanced.tree.cubic_bounds is cubic


def test_an_explicit_galaxy_preset_refuses_what_the_runner_does_not_run(fused_lane):
    state, masses = _system()
    state = jnp.asarray(state, jnp.float32)
    masses = jnp.asarray(masses, jnp.float32)
    for cfg, kw in (
        (_config(fixed_timestep=False), {}),
        (_config(fmm_refresh_every=2), {}),
        (_config(), dict(active_indices_schedule=jnp.zeros((6, 4), jnp.int32))),
    ):
        with pytest.raises(NotImplementedError, match="runner lane"):
            integrate(state, masses, cfg, _params(), **kw)


def test_the_runner_lane_needs_a_gpu_backend(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    state, masses = _system()
    with pytest.raises(NotImplementedError, match="GPU backend"):
        integrate(
            jnp.asarray(state, jnp.float32),
            jnp.asarray(masses, jnp.float32),
            _config(),
            _params(),
        )


# --------------------------------------------------------------------------
# Integrating tests (CPU, the fused lane's pure-JAX route).
# --------------------------------------------------------------------------


def _jaccpot_runner_reference(
    state, masses, steps, params, *, preset_mac=False, use_pallas=False
):
    """``ProductionRunner`` called directly: a solver from jaccpot's API, the halo as
    a closure over Odisseo's potential, its parameters as float32 arrays. The
    geometric MAC (``SMALL_N``'s), or with ``preset_mac`` the preset's own."""
    from jaccpot import (
        FastMultipoleMethod,
        ForceParams,
        ProductionRunner,
        SegmentConfig,
    )

    from odisseo.potentials import combined_external_acceleration_vmpa_switch

    base = FastMultipoleMethod(preset="galaxy").advanced
    solver = FastMultipoleMethod(
        preset="galaxy",
        theta=0.6,
        G=1.0,
        softening=SOFTENING,
        use_pallas=use_pallas,
        advanced=dataclasses.replace(
            base,
            tree=dataclasses.replace(base.tree, leaf_target=16, cell_min_level=None),
            runtime=dataclasses.replace(base.runtime, fixed_order=4),
            mac_type=base.mac_type if preset_mac else "dehnen",
        ),
    )
    criterion = solver._impl.mac_type == "dehnen_error"
    halo_config = SimulationConfig(external_accelerations=(NFW_POTENTIAL,))

    def halo(state_in, p):
        return combined_external_acceleration_vmpa_switch(state_in, halo_config, p)

    runner = ProductionRunner(
        solver,
        settings=SMALL_N_PRESET_MAC if preset_mac else SMALL_N,
        segment=SegmentConfig(steps=50, diagnostics_every=0),
        external_acceleration_fn=halo,
    )
    p32 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), params)
    res = runner.run(
        state,
        masses,
        dt=DT,
        num_steps=steps,
        # dehnen_error takes the solver's own G and softening only
        force_params=None if criterion else ForceParams(G=1.0, eps2=SOFTENING**2),
        external_params=p32,
    )
    return np.asarray(res.state)


def _strict_programs(solver):
    cache = solver._impl._strict_fused_jit_function_cache
    return {k for k in cache if str(k[0]).startswith("strict_velocity_verlet")}


def test_integrate_is_the_jaccpot_runner_and_a_second_call_compiles_nothing(fused_lane):
    state_np, masses_np = _system()
    state = jnp.asarray(state_np, jnp.float32)
    masses = jnp.asarray(masses_np, jnp.float32)
    cfg = _config()

    out, lane = integrate(state, masses, cfg, _params(), return_lane=True)
    assert lane == "fmm_runner"
    first = np.asarray(out)
    ref = _jaccpot_runner_reference(state, masses, 6, _params())
    assert np.array_equal(first, ref)
    assert not state.is_deleted()  # the caller's state is kept

    # another dt, G and halo: the kept solver, no new program in its cache
    ((solver, _),) = rc._CACHE.values()
    programs = _strict_programs(solver)
    assert programs
    second = integrate(state, masses, cfg, _params(steps=12, G=1.1, mvir=12.0))
    ((solver_2, _),) = rc._CACHE.values()
    assert solver_2 is solver
    assert _strict_programs(solver) == programs
    assert not np.array_equal(np.asarray(second), first)  # the values took effect
    # and the first values again: the first call's state
    assert np.array_equal(np.asarray(integrate(state, masses, cfg, _params())), first)


def test_the_default_mac_is_the_presets_and_runs_as_jaccpots_runner(
    fused_lane, monkeypatch
):
    """Without a MAC of its own (``fmm_mac_type="dehnen"``, Odisseo's default, keeps
    the settings') the lane runs jaccpot's preset MAC -- ``dehnen_error`` at
    ``adaptive_eps`` 1e-4 since jaccpot's 2026-10-21 preset change, the geometric
    MAC before -- and computes what jaccpot's runner on ``preset="galaxy"`` with
    that MAC computes, bit for bit. Under ``dehnen_error`` G is the solver's, so a
    call with another G builds another solver; on the geometric MAC the kept one
    serves it."""
    for key, val in _CSR_ON_CPU.items():
        monkeypatch.setenv(key, val)
    state_np, masses_np = _system()
    state = jnp.asarray(state_np, jnp.float32)
    masses = jnp.asarray(masses_np, jnp.float32)
    cfg = _config(fmm_use_pallas=True, fmm_galaxy_settings=SMALL_N_PRESET_MAC)

    out = integrate(state, masses, cfg, _params())
    ((solver, _),) = rc._CACHE.values()
    impl = solver._impl
    criterion = GALAXY_SETTINGS.mac_type == "dehnen_error"
    assert impl.mac_type == GALAXY_SETTINGS.mac_type
    assert impl._flat_walk_criterion_active() == criterion
    if criterion:
        assert impl.adaptive_eps == GALAXY_SETTINGS.adaptive_eps
    ref = _jaccpot_runner_reference(
        state, masses, 6, _params(), preset_mac=True, use_pallas=True
    )
    assert np.array_equal(np.asarray(out), ref)

    # the kept solver a call with another G would run on (looked up, not run)
    settings = rc.galaxy_settings(cfg, state.dtype)
    solver_2, _ = rc._solver_and_runner(cfg, _params(G=1.1), settings, N, True)
    assert (solver_2 is solver) == (not criterion)
    rc.clear_runner_cache()


def test_history_runs_return_every_kth_state_with_the_diagnostics(fused_lane):
    """6 steps, 3 snapshots: the states at steps 0, 2 and 4 (2-step segments), at
    times 0, 2 dt, 4 dt, bitwise the plain runs of 2 and 4 steps in segments of 2;
    the energy (the runner's, at every edge) and angular momentum come along."""
    from jaccpot import SegmentConfig

    state_np, masses_np = _system()
    state = jnp.asarray(state_np, jnp.float32)
    masses = jnp.asarray(masses_np, jnp.float32)
    snaps, lane = integrate(
        state,
        masses,
        _config(return_snapshots=True, num_snapshots=3),
        _params(),
        return_lane=True,
    )
    assert lane == "fmm_runner"
    # the kept runner keeps no snapshot (nor the collector) for the next call
    ((_, runner),) = rc._CACHE.values()
    assert runner.callbacks == () and runner.step_callback is None
    assert isinstance(snaps.states, np.ndarray) and snaps.states.shape == (3, N, 2, 3)
    np.testing.assert_array_equal(snaps.times, [0.0, 2 * DT, 4 * DT])
    assert np.array_equal(snaps.states[0], np.asarray(state))
    for k, steps in ((1, 2), (2, 4)):
        plain = rc.integrate_runner_jaccpot(
            state,
            masses,
            _config(num_timesteps=steps),
            _params(steps=steps),
            segment=SegmentConfig(steps=2, diagnostics_every=0),
        )
        assert np.array_equal(snaps.states[k], np.asarray(plain.state)), k
    assert snaps.num_iterations == 6
    assert np.all(np.isfinite(snaps.total_energy))
    assert np.all(np.isfinite(snaps.angular_momentum))
    # energy: conserved to the integrator's and the fp32 force's level over 6 steps
    drift = np.abs(snaps.total_energy - snaps.total_energy[0]) / abs(
        snaps.total_energy[0]
    )
    assert np.max(drift) < 1e-4
    # angular momentum at step 0 is the state's, in fp64
    x, v = state_np[:, 0], state_np[:, 1]
    expected = np.sum(masses_np[:, None] * np.cross(x, v), axis=0)
    np.testing.assert_allclose(
        snaps.angular_momentum[0], expected, rtol=1e-6, atol=1e-9
    )


@pytest.mark.slow
def test_the_runner_lane_follows_the_direct_lane_as_the_coupler_does(monkeypatch):
    """At p8, theta 0.3 the fused lane's self-gravity is at the fp32 floor of the
    fp64 direct sum (rel-L2 5.4e-7 in this draw), so 6 steps with the halo stay with
    Odisseo's fp64 direct lane (same velocity Verlet, same kernel, same G and
    softening) as closely as the coupler's CPU lane (preset "fast") does at the same
    order and angle: the units and the kernel reach jaccpot as on the coupler."""
    state_np, masses_np = _system()
    params = _params()
    direct = np.asarray(
        integrate(
            jnp.asarray(state_np),
            jnp.asarray(masses_np),
            _config(acceleration_scheme=DIRECT_ACC),
            params,
        )
    )
    s32 = jnp.asarray(state_np, jnp.float32)
    m32 = jnp.asarray(masses_np, jnp.float32)
    coupler, lane = integrate(
        s32,
        m32,
        _config(fmm_preset="fast", fmm_leaf_size=16, fmm_max_order=8, fmm_theta=0.3),
        params,
        return_lane=True,
    )
    assert lane == "fmm_forward"
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    runner, lane = integrate(
        s32,
        m32,
        _config(
            fmm_galaxy_settings=dataclasses.replace(SMALL_N, max_order=8, theta=0.3)
        ),
        params,
        return_lane=True,
    )
    assert lane == "fmm_runner"

    change = np.linalg.norm(direct[:, 1] - state_np[:, 1])

    def dv_error(x):
        return np.linalg.norm(np.asarray(x, np.float64)[:, 1] - direct[:, 1]) / change

    def x_error(x):
        return np.linalg.norm(np.asarray(x, np.float64)[:, 0] - direct[:, 0]) / (
            np.linalg.norm(direct[:, 0])
        )

    assert dv_error(runner) < 1e-4
    assert x_error(runner) < 1e-6
    assert dv_error(coupler) < 1e-4
    assert x_error(coupler) < 1e-6
