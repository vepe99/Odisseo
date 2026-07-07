import jax.numpy as jnp

from odisseo.option_classes import FMM_ACC, SimulationConfig, SimulationParams


def test_fixed_uses_core_kernel_scaffold_when_enabled(monkeypatch):
    from odisseo import jaccpot_coupling as jc

    state = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    mass = jnp.ones((4,), dtype=jnp.float32)
    cfg = SimulationConfig(
        N_particles=4,
        fixed_timestep=True,
        num_timesteps=3,
        acceleration_scheme=FMM_ACC,
        fmm_tree_build_mode="static_radix",
        fmm_preset="fast",
        fmm_runtime_path="auto",
        external_accelerations=(),
    )
    params = SimulationParams(G=1.0, t_end=1e-3)

    calls = {"core": 0}

    def _fake_build_core(*args, **kwargs):
        del args, kwargs

        def _kernel(y, m, prepared_state):
            del m
            calls["core"] += 1
            next_state = y.at[:, 1, :].set(y[:, 1, :] + 1.0)
            return jc.JaccpotCoreKernelOutput(
                next_state=next_state,
                acceleration=jnp.zeros_like(y[:, 0, :]),
                prepared_state={"step": calls["core"], "prev": prepared_state},
                execute_count=1,
                prepare_count=1 if calls["core"] == 1 else 0,
                refresh_count=0 if calls["core"] == 1 else 1,
            )

        return _kernel, jc.JaccpotCoreKernelConfig(
            mode="fixed_step_update",
            leaf_size=16,
            max_order=2,
            preset="fast",
            runtime_path="auto",
            tree_build_mode="static_radix",
        )

    monkeypatch.setenv("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "1")
    monkeypatch.setattr(jc, "build_compiled_jaccpot_core_kernel", _fake_build_core)

    out = jc.integrate_leapfrog_jaccpot_active(
        state,
        mass,
        cfg,
        params,
        num_steps=3,
        leaf_size=16,
        max_order=2,
        fmm_preset="fast",
        fmm_runtime_path="auto",
        fmm_tree_build_mode="static_radix",
        return_history=False,
    )

    assert out.shape == state.shape
    assert calls["core"] == 3
    assert jnp.allclose(out[:, 1, :], 3.0)

