import types

import jax.numpy as jnp

from odisseo.option_classes import (
    DOPRI5,
    FMM_ACC,
    SimulationConfig,
    SimulationParams,
)


class LargeNPreparedState:
    def __init__(self, token):
        self.token = int(token)



def test_adaptive_uses_core_kernel_scaffold_when_enabled(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        fixed_timestep=False,
        num_timesteps=2,
        acceleration_scheme=FMM_ACC,
        diffrax_solver=DOPRI5,
        fmm_tree_build_mode="static_radix",
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
    )
    params = SimulationParams(G=1.0, t_end=1e-3)

    calls = {"core": 0}
    seen_prepared_inputs = []

    class _FakeSolver:
        def prepare_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("core-kernel path should bypass direct prepare_state")

        def evaluate_prepared_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "core-kernel path should bypass direct evaluate_prepared_state"
            )

    def _fake_build_solver(**kwargs):
        del kwargs
        return _FakeSolver()

    def _fake_build_core(*args, **kwargs):
        del args, kwargs

        def _kernel(y, m, prepared_state, *, refresh_prepared=True):
            del m, refresh_prepared
            calls["core"] += 1
            seen_prepared_inputs.append(prepared_state)
            return jc.JaccpotCoreKernelOutput(
                next_state=y,
                acceleration=jnp.zeros_like(y[:, 0, :]),
                prepared_state=LargeNPreparedState(calls["core"]),
                execute_count=1,
                prepare_count=1,
                refresh_count=0 if calls["core"] == 1 else 1,
            )

        return _kernel, jc.JaccpotCoreKernelConfig(
            mode="rhs_only",
            leaf_size=16,
            max_order=2,
            preset="large_n_gpu",
            runtime_path="large_n",
            tree_build_mode="static_radix",
        )

    def _fake_diffeqsolve(
        *,
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat,
        stepsize_controller,
        max_steps,
    ):
        del solver, t0, t1, dt0, saveat, stepsize_controller, max_steps
        _ = terms.vf(0.0, y0, None)
        _ = terms.vf(0.5, y0, None)
        return types.SimpleNamespace(
            ys=jnp.stack((y0, y0), axis=0),
            ts=jnp.asarray([0.0, 1.0]),
            stats={"num_accepted_steps": 1, "num_rejected_steps": 0, "num_steps": 2},
        )

    monkeypatch.setenv("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "1")
    monkeypatch.setenv("ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE", "python")
    monkeypatch.setattr(jc, "_build_fmm_solver", _fake_build_solver)
    monkeypatch.setattr(jc, "build_compiled_jaccpot_core_kernel", _fake_build_core)
    monkeypatch.setattr(jc.diffrax, "diffeqsolve", _fake_diffeqsolve)

    out = jc.integrate_diffrax_jaccpot_active(
        state,
        mass,
        cfg,
        params,
        num_steps=2,
        leaf_size=16,
        max_order=2,
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
        fmm_tree_build_mode="static_radix",
    )

    assert out.shape == state.shape
    assert calls["core"] == 2
    assert seen_prepared_inputs[0] is None
    assert getattr(seen_prepared_inputs[1], "token", None) == 1


def test_adaptive_core_kernel_defaults_to_no_python_prepared_cache(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        fixed_timestep=False,
        num_timesteps=2,
        acceleration_scheme=FMM_ACC,
        diffrax_solver=DOPRI5,
        fmm_tree_build_mode="static_radix",
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
    )
    params = SimulationParams(G=1.0, t_end=1e-3)

    calls = {"core": 0}
    seen_prepared_inputs = []

    class _FakeSolver:
        def prepare_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("core-kernel path should bypass direct prepare_state")

        def evaluate_prepared_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "core-kernel path should bypass direct evaluate_prepared_state"
            )

    def _fake_build_solver(**kwargs):
        del kwargs
        return _FakeSolver()

    def _fake_build_core(*args, **kwargs):
        del args, kwargs

        def _kernel(y, m, prepared_state, *, refresh_prepared=True):
            del m, refresh_prepared
            calls["core"] += 1
            seen_prepared_inputs.append(prepared_state)
            return jc.JaccpotCoreKernelOutput(
                next_state=y,
                acceleration=jnp.zeros_like(y[:, 0, :]),
                prepared_state=LargeNPreparedState(calls["core"]),
                execute_count=1,
                prepare_count=1,
                refresh_count=0 if calls["core"] == 1 else 1,
            )

        return _kernel, jc.JaccpotCoreKernelConfig(
            mode="rhs_only",
            leaf_size=16,
            max_order=2,
            preset="large_n_gpu",
            runtime_path="large_n",
            tree_build_mode="static_radix",
        )

    def _fake_diffeqsolve(
        *,
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat,
        stepsize_controller,
        max_steps,
    ):
        del solver, t0, t1, dt0, saveat, stepsize_controller, max_steps
        _ = terms.vf(0.0, y0, None)
        _ = terms.vf(0.5, y0, None)
        return types.SimpleNamespace(
            ys=jnp.stack((y0, y0), axis=0),
            ts=jnp.asarray([0.0, 1.0]),
            stats={"num_accepted_steps": 1, "num_rejected_steps": 0, "num_steps": 2},
        )

    monkeypatch.setenv("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "1")
    monkeypatch.delenv("ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE", raising=False)
    monkeypatch.setattr(jc, "_build_fmm_solver", _fake_build_solver)
    monkeypatch.setattr(jc, "build_compiled_jaccpot_core_kernel", _fake_build_core)
    monkeypatch.setattr(jc.diffrax, "diffeqsolve", _fake_diffeqsolve)

    timing_stats = {}
    out = jc.integrate_diffrax_jaccpot_active(
        state,
        mass,
        cfg,
        params,
        num_steps=2,
        leaf_size=16,
        max_order=2,
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
        fmm_tree_build_mode="static_radix",
        timing_stats=timing_stats,
    )

    assert out.shape == state.shape
    assert calls["core"] == 2
    assert seen_prepared_inputs == [None, None]
    assert timing_stats["adaptive_prepared_cache_mode"] == "none"
    assert timing_stats["adaptive_python_prepared_cache_enabled"] is False



def test_adaptive_core_kernel_refresh_cadence_rhs_calls(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        fixed_timestep=False,
        num_timesteps=3,
        acceleration_scheme=FMM_ACC,
        diffrax_solver=DOPRI5,
        fmm_tree_build_mode="static_radix",
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
        fmm_adaptive_refresh_rhs_calls=2,
    )
    params = SimulationParams(G=1.0, t_end=1e-3)

    calls = {"core": 0}
    seen_prepared_inputs = []
    seen_refresh_flags = []

    class _FakeSolver:
        def prepare_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("core-kernel path should bypass direct prepare_state")

        def evaluate_prepared_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "core-kernel path should bypass direct evaluate_prepared_state"
            )

    def _fake_build_solver(**kwargs):
        del kwargs
        return _FakeSolver()

    def _fake_build_core(*args, **kwargs):
        del args, kwargs

        def _kernel(y, m, prepared_state, *, refresh_prepared=True):
            del m
            calls["core"] += 1
            seen_prepared_inputs.append(prepared_state)
            seen_refresh_flags.append(bool(refresh_prepared))
            return jc.JaccpotCoreKernelOutput(
                next_state=y,
                acceleration=jnp.zeros_like(y[:, 0, :]),
                prepared_state=LargeNPreparedState(calls["core"]),
                execute_count=1,
                prepare_count=1 if refresh_prepared else 0,
                refresh_count=0,
            )

        return _kernel, jc.JaccpotCoreKernelConfig(
            mode="rhs_only",
            leaf_size=16,
            max_order=2,
            preset="large_n_gpu",
            runtime_path="large_n",
            tree_build_mode="static_radix",
        )

    def _fake_diffeqsolve(
        *,
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat,
        stepsize_controller,
        max_steps,
    ):
        del solver, t0, t1, dt0, saveat, stepsize_controller, max_steps
        _ = terms.vf(0.0, y0, None)
        _ = terms.vf(0.5, y0, None)
        _ = terms.vf(0.75, y0, None)
        return types.SimpleNamespace(
            ys=jnp.stack((y0, y0), axis=0),
            ts=jnp.asarray([0.0, 1.0]),
            stats={"num_accepted_steps": 1, "num_rejected_steps": 0, "num_steps": 3},
        )

    monkeypatch.setenv("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "1")
    monkeypatch.setenv("ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE", "python")
    monkeypatch.setattr(jc, "_build_fmm_solver", _fake_build_solver)
    monkeypatch.setattr(jc, "build_compiled_jaccpot_core_kernel", _fake_build_core)
    monkeypatch.setattr(jc.diffrax, "diffeqsolve", _fake_diffeqsolve)

    timing_stats = {}
    out = jc.integrate_diffrax_jaccpot_active(
        state,
        mass,
        cfg,
        params,
        num_steps=3,
        leaf_size=16,
        max_order=2,
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
        fmm_tree_build_mode="static_radix",
        timing_stats=timing_stats,
    )

    assert out.shape == state.shape
    assert calls["core"] == 3
    assert seen_prepared_inputs[0] is None
    assert getattr(seen_prepared_inputs[1], "token", None) == 1
    assert getattr(seen_prepared_inputs[2], "token", None) == 2
    assert seen_refresh_flags == [True, False, True]
    assert timing_stats["adaptive_refresh_rhs_calls_target"] == 2
    assert timing_stats["adaptive_core_refresh_cadence_skips_rhs_calls"] == 1


def test_adaptive_core_kernel_refresh_cadence_displacement_threshold(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        fixed_timestep=False,
        num_timesteps=3,
        acceleration_scheme=FMM_ACC,
        diffrax_solver=DOPRI5,
        fmm_tree_build_mode="static_radix",
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
        fmm_adaptive_refresh_rhs_calls=1,
        fmm_adaptive_refresh_displacement_threshold=1.0,
    )
    params = SimulationParams(G=1.0, t_end=1e-3)

    calls = {"core": 0}
    seen_prepared_inputs = []
    seen_refresh_flags = []

    class _FakeSolver:
        def prepare_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("core-kernel path should bypass direct prepare_state")

        def evaluate_prepared_state(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "core-kernel path should bypass direct evaluate_prepared_state"
            )

    def _fake_build_solver(**kwargs):
        del kwargs
        return _FakeSolver()

    def _fake_build_core(*args, **kwargs):
        del args, kwargs

        def _kernel(y, m, prepared_state, *, refresh_prepared=True):
            del m
            calls["core"] += 1
            seen_prepared_inputs.append(prepared_state)
            seen_refresh_flags.append(bool(refresh_prepared))
            return jc.JaccpotCoreKernelOutput(
                next_state=y,
                acceleration=jnp.zeros_like(y[:, 0, :]),
                prepared_state=LargeNPreparedState(calls["core"]),
                execute_count=1,
                prepare_count=1 if refresh_prepared else 0,
                refresh_count=0,
            )

        return _kernel, jc.JaccpotCoreKernelConfig(
            mode="rhs_only",
            leaf_size=16,
            max_order=2,
            preset="large_n_gpu",
            runtime_path="large_n",
            tree_build_mode="static_radix",
        )

    def _fake_diffeqsolve(
        *,
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat,
        stepsize_controller,
        max_steps,
    ):
        del solver, t0, t1, dt0, saveat, stepsize_controller, max_steps
        y1 = y0.at[:, 0, :].set(0.25)
        y2 = y0.at[:, 0, :].set(2.0)
        _ = terms.vf(0.0, y0, None)
        _ = terms.vf(0.5, y1, None)
        _ = terms.vf(0.75, y2, None)
        return types.SimpleNamespace(
            ys=jnp.stack((y0, y0), axis=0),
            ts=jnp.asarray([0.0, 1.0]),
            stats={"num_accepted_steps": 1, "num_rejected_steps": 0, "num_steps": 3},
        )

    monkeypatch.setenv("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "1")
    monkeypatch.setenv("ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE", "python")
    monkeypatch.setattr(jc, "_build_fmm_solver", _fake_build_solver)
    monkeypatch.setattr(jc, "build_compiled_jaccpot_core_kernel", _fake_build_core)
    monkeypatch.setattr(jc.diffrax, "diffeqsolve", _fake_diffeqsolve)

    timing_stats = {}
    out = jc.integrate_diffrax_jaccpot_active(
        state,
        mass,
        cfg,
        params,
        num_steps=3,
        leaf_size=16,
        max_order=2,
        fmm_preset="large_n_gpu",
        fmm_runtime_path="large_n",
        fmm_tree_build_mode="static_radix",
        timing_stats=timing_stats,
    )

    assert out.shape == state.shape
    assert calls["core"] == 3
    assert seen_prepared_inputs[0] is None
    assert getattr(seen_prepared_inputs[1], "token", None) == 1
    assert getattr(seen_prepared_inputs[2], "token", None) == 2
    assert seen_refresh_flags == [True, False, True]
    assert timing_stats["adaptive_refresh_displacement_threshold"] == 1.0
    assert timing_stats["adaptive_core_refresh_cadence_skips_displacement"] == 1
