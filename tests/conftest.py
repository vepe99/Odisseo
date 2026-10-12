import jax
import jax.numpy as jnp
import pytest

# The jaccpot lanes require jax_enable_x64 (README), and nothing switches it on any
# more: until 2026-10 yggdrax did, when the first FMM call imported it. A full run had
# x64 on from collection anyway (test_blockstep_fmm.py, the first module, enables it
# at import); a single file ran with it off until its first FMM call. Turning it on
# here, before any test module is imported, gives every run the full suite's state.
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def two_body_state():
    state = jnp.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    ])
    mass = jnp.array([1.0, 1.0])
    return state, mass