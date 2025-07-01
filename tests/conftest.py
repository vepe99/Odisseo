import pytest
import jax.numpy as jnp

@pytest.fixture
def two_body_state():
    state = jnp.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    ])
    mass = jnp.array([1.0, 1.0])
    return state, mass