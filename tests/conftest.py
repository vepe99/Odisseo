import os

# XLA:CPU's LLVM loop vectorizer can stall a compile for many minutes on jax 0.11.2
# (a recursion in `llvm::vputils::onlyFirstLaneUsed`): jaccpot's fused lane, which
# tests/test_runner_lane.py runs on CPU, takes 45+ min to compile with it. jaccpot's
# own conftest turns it off for the same reason. It must be in XLA_FLAGS before the
# backend starts, so it goes before anything creates an array.
if "xla_backend_extra_options" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_backend_extra_options=-vectorize-loops=false"
    ).strip()

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