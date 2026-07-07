import jax.numpy as jnp
import pytest

from odisseo.option_classes import SimulationConfig, SimulationParams


class _FakeSolver:
    def __init__(self):
        self.prepare_calls = 0
        self.evaluate_calls = 0
        self.refresh_calls = 0

    def prepare_state(self, positions, masses, *, leaf_size, max_order):
        self.prepare_calls += 1
        return {
            "positions": positions,
            "masses": masses,
            "leaf_size": leaf_size,
            "max_order": max_order,
        }

    def evaluate_prepared_state(self, prepared, *, target_indices, return_potential):
        self.evaluate_calls += 1
        assert target_indices is None
        assert return_potential is False
        return jnp.ones_like(prepared["positions"]) * 2.0

    def refresh_prepared_state(
        self,
        prepared_state,
        positions,
        masses,
        *,
        leaf_size,
        max_order,
    ):
        self.refresh_calls += 1
        return {
            "positions": positions,
            "masses": masses,
            "leaf_size": leaf_size,
            "max_order": max_order,
            "refreshed": True,
            "prev": prepared_state,
        }


def _base_config():
    return SimulationConfig(
        N_particles=4,
        fixed_timestep=True,
        fmm_tree_build_mode="static_radix",
    )


def _base_state_mass():
    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    return state, mass


def test_core_kernel_rhs_only_scaffold(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    fake_solver = _FakeSolver()
    monkeypatch.setattr(jc, "_build_fmm_solver", lambda **_: fake_solver)

    kernel, meta = jc.build_compiled_jaccpot_core_kernel(
        _base_config(),
        SimulationParams(G=1.0, t_end=1.0),
        mode="rhs_only",
        leaf_size=16,
        max_order=4,
    )
    state, mass = _base_state_mass()
    out = kernel(state, mass)

    assert meta.mode == "rhs_only"
    assert out.next_state.shape == state.shape
    assert out.acceleration.shape == (4, 3)
    assert out.execute_count == 1
    assert out.prepare_count == 1
    assert out.refresh_count == 0
    assert fake_solver.prepare_calls == 1
    assert fake_solver.evaluate_calls == 1


def test_core_kernel_fixed_step_update_scaffold(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    fake_solver = _FakeSolver()
    monkeypatch.setattr(jc, "_build_fmm_solver", lambda **_: fake_solver)

    kernel, meta = jc.build_compiled_jaccpot_core_kernel(
        _base_config(),
        SimulationParams(G=1.0, t_end=1.0),
        mode="fixed_step_update",
        dt=0.5,
        leaf_size=16,
        max_order=4,
    )
    state, mass = _base_state_mass()
    out = kernel(state, mass)

    assert meta.mode == "fixed_step_update"
    assert out.next_state.shape == state.shape
    # Zero initial velocity + acceleration=2 means v_next = 1.0
    assert jnp.allclose(out.next_state[:, 1, :], 1.0)
    assert fake_solver.prepare_calls == 1
    assert fake_solver.evaluate_calls == 1


def test_core_kernel_invalid_mode():
    from odisseo import jaccpot_coupling as jc

    with pytest.raises(ValueError, match="mode must be one of"):
        jc.build_compiled_jaccpot_core_kernel(
            _base_config(),
            SimulationParams(G=1.0, t_end=1.0),
            mode="unknown_mode",
        )


def test_core_kernel_reuses_solver_instance(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    calls = {"build_solver": 0}

    def _build_solver(**_kwargs):
        calls["build_solver"] += 1
        return _FakeSolver()

    monkeypatch.setattr(jc, "_build_fmm_solver", _build_solver)

    kernel, _meta = jc.build_compiled_jaccpot_core_kernel(
        _base_config(),
        SimulationParams(G=1.0, t_end=1.0),
        mode="rhs_only",
    )
    state, mass = _base_state_mass()
    _ = kernel(state, mass)
    _ = kernel(state, mass)
    assert calls["build_solver"] == 1


def test_core_kernel_prefers_refresh_when_prepared_state_present(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    fake_solver = _FakeSolver()
    monkeypatch.setattr(jc, "_build_fmm_solver", lambda **_: fake_solver)

    kernel, _meta = jc.build_compiled_jaccpot_core_kernel(
        _base_config(),
        SimulationParams(G=1.0, t_end=1.0),
        mode="rhs_only",
        leaf_size=16,
        max_order=4,
    )
    state, mass = _base_state_mass()

    # The scaffold only attempts an in-place refresh when the prepared state is a
    # LargeNPreparedState (guarded by type name in jaccpot_coupling). Mimic that
    # so the refresh-preference branch is exercised rather than a full prepare.
    class LargeNPreparedState(dict):
        pass

    out = kernel(state, mass, prepared_state=LargeNPreparedState(dummy=True))
    assert out.prepare_count == 0
    assert out.refresh_count == 1
    assert fake_solver.refresh_calls == 1
    assert fake_solver.prepare_calls == 0
