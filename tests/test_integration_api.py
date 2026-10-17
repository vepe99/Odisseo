import jax.numpy as jnp

from odisseo.option_classes import (
    DIRECT_ACC,
    FMM_ACC,
    SimulationConfig,
    SimulationParams,
)


def test_integrate_dispatches_to_direct_time_integration(monkeypatch):
    from odisseo import integration_api as api

    calls = {"direct": 0}

    def fake_time_integration(state, mass, config, params):
        calls["direct"] += 1
        return "direct-path"

    monkeypatch.setattr(api, "time_integration", fake_time_integration)

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        acceleration_scheme=DIRECT_ACC,
        fixed_timestep=True,
    )
    params = SimulationParams(G=1.0, t_end=1.0)

    out = api.integrate(state, mass, cfg, params)
    assert out == "direct-path"
    assert calls["direct"] == 1


def test_integrate_dispatches_to_fmm_coupler(monkeypatch):
    from odisseo import integration_api as api

    calls = {"fmm": 0}

    def fake_fmm(*args, **kwargs):
        calls["fmm"] += 1
        assert kwargs["num_steps"] == 7
        assert kwargs["refresh_every"] == 3
        assert kwargs["leaf_size"] == 24
        assert kwargs["max_order"] == 5
        assert kwargs["fmm_preset"] == "accurate"
        assert kwargs["fmm_basis"] == "real"
        assert kwargs["fmm_theta"] == 0.4
        assert kwargs["fmm_runtime_path"] == "auto"
        assert kwargs["fmm_working_dtype"] == state.dtype
        assert kwargs["fmm_mac_type"] == "bh"
        assert kwargs["fmm_farfield_mode"] == "pair_grouped"
        assert kwargs["fmm_nearfield_mode"] == "bucketed"
        assert kwargs["fmm_nearfield_edge_chunk_size"] == 96
        assert kwargs["fmm_tree_leaf_target"] == 20
        assert kwargs["enforce_static_shape_contract"] is True
        assert kwargs["static_shape_warmup_prepares"] == 2
        assert kwargs["rematerialize_between_refresh"] is False
        assert kwargs["return_history"] is False
        return jnp.zeros((4, 2, 3), dtype=jnp.float32)

    monkeypatch.setattr(api, "integrate_leapfrog_jaccpot_active", fake_fmm)

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        return_snapshots=False,
        num_timesteps=7,
        fmm_refresh_every=3,
        fmm_leaf_size=24,
        fmm_max_order=5,
        fmm_preset="accurate",
        fmm_basis="real",
        fmm_theta=0.4,
        fmm_mac_type="bh",
        # Values jaccpot accepts that differ from the defaults, so the plumbing
        # check still means something ("dense"/"pairwise" are not jaccpot modes).
        fmm_farfield_mode="pair_grouped",
        fmm_nearfield_mode="bucketed",
        fmm_nearfield_edge_chunk_size=96,
        fmm_tree_leaf_target=20,
        fmm_enforce_static_shape_contract=True,
        fmm_static_shape_warmup_prepares=2,
        fmm_rematerialize_between_refresh=False,
    )
    params = SimulationParams(G=1.0, t_end=1.0)

    out = api.integrate(state, mass, cfg, params)
    assert out.shape == state.shape
    assert calls["fmm"] == 1


def test_integrate_auto_selects_large_n_gpu_profile(monkeypatch):
    # The auto profile picks jaccpot's runner lane where the runner runs the call
    # (tests/test_runner_lane.py); for one it does not -- here an active-particle
    # schedule -- it keeps the coupler on the large_n_gpu profile, as before.
    from odisseo import integration_api as api

    calls = {"fmm": 0}

    def fake_fmm(*args, **kwargs):
        calls["fmm"] += 1
        assert kwargs["fmm_preset"] == "large_n_gpu"
        assert kwargs["fmm_runtime_path"] == "large_n"
        assert kwargs["fmm_working_dtype"] == jnp.float32
        return jnp.zeros((6, 2, 3), dtype=jnp.float32)

    monkeypatch.setattr(api, "integrate_leapfrog_jaccpot_active", fake_fmm)
    monkeypatch.setattr(api.jax, "default_backend", lambda: "gpu")

    state = jnp.zeros((6, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((6,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=6,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        num_timesteps=3,
        fmm_preset="fast",
        fmm_auto_large_n_profile=True,
        fmm_large_n_min_particles=6,
        fmm_runtime_path="auto",
        fmm_large_n_force_fp32=True,
    )
    params = SimulationParams(G=1.0, t_end=1.0)

    out, lane = api.integrate(
        state,
        mass,
        cfg,
        params,
        active_indices_schedule=jnp.zeros((3, 6), dtype=jnp.int32),
        return_lane=True,
    )
    assert out.shape == state.shape
    assert calls["fmm"] == 1
    assert lane == "fmm_forward"


def test_integrate_fmm_requires_fixed_timestep():
    from odisseo import integration_api as api

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    calls = {"adaptive": 0}

    def fake_adaptive(*args, **kwargs):
        calls["adaptive"] += 1
        assert kwargs["num_steps"] == 4
        return jnp.zeros_like(state)

    api.integrate_diffrax_jaccpot_active = fake_adaptive
    cfg = SimulationConfig(
        N_particles=4,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=False,
        num_timesteps=4,
    )
    params = SimulationParams(G=1.0, t_end=1.0)
    out = api.integrate(state, mass, cfg, params)
    assert out.shape == state.shape
    assert calls["adaptive"] == 1


def _history(first, last, n=4):
    """A fake history: the state after step s filled with s, s = first .. last."""
    return jnp.stack(
        [
            jnp.full((n, 2, 3), float(s), dtype=jnp.float32)
            for s in range(first, last + 1)
        ]
    )


def test_fmm_snapshots_are_every_kth_state_from_the_initial_one(monkeypatch):
    # The fixed-step FMM lanes package their history as the direct lane does: the
    # state after k * K steps (K = num_timesteps // num_snapshots), the initial
    # state first, at times k * K * dt. The coupler's history starts after step 1,
    # and its snapshots used to be steps 1, 1 + K, ... labelled linspace(0, t_end).
    from odisseo import differentiable
    from odisseo import integration_api as api

    def fake_coupler(*args, **kwargs):
        assert kwargs["return_history"] is True
        return _history(1, kwargs["num_steps"])

    def fake_differentiable(state, mass, config, params, **kwargs):
        assert kwargs["return_history"] is True
        return _history(0, int(config.num_timesteps))

    monkeypatch.setattr(api, "integrate_leapfrog_jaccpot_active", fake_coupler)
    monkeypatch.setattr(
        differentiable, "integrate_leapfrog_differentiable", fake_differentiable
    )
    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    params = SimulationParams(G=1.0, t_end=0.6)
    for flags in ({}, {"fmm_differentiable": True}):
        cfg = SimulationConfig(
            N_particles=4,
            acceleration_scheme=FMM_ACC,
            fixed_timestep=True,
            num_timesteps=6,
            return_snapshots=True,
            num_snapshots=3,
            **flags,
        )
        out = api.integrate(state, mass, cfg, params)
        assert out.states.shape == (3, 4, 2, 3)
        assert [float(s) for s in out.states[:, 0, 0, 0]] == [0.0, 2.0, 4.0]
        assert jnp.allclose(out.times, jnp.asarray([0.0, 0.2, 0.4]))
