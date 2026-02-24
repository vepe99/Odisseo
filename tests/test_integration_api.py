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
    )
    params = SimulationParams(G=1.0, t_end=1.0)

    out = api.integrate(state, mass, cfg, params)
    assert out.shape == state.shape
    assert calls["fmm"] == 1


def test_integrate_fmm_requires_fixed_timestep():
    from odisseo.integration_api import integrate

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=False,
        num_timesteps=4,
    )
    params = SimulationParams(G=1.0, t_end=1.0)

    try:
        integrate(state, mass, cfg, params)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Expected NotImplementedError for fixed_timestep=False")
