import pytest
import jax.numpy as jnp
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.dynamics import direct_acc, direct_acc_matrix, direct_acc_laxmap, direct_acc_for_loop


@pytest.mark.parametrize("force_func", [
    direct_acc,
    direct_acc_matrix,
    direct_acc_laxmap,
    direct_acc_for_loop
])
def test_full_pairwise_force(force_func, ):
    config = SimulationConfig(
        N_particles=2,
        return_snapshots=False,)
    params = SimulationParams(G=1,)

    # Simple 2-body system with analytical solution
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    vel = jnp.zeros_like(pos)
    mass = jnp.array([1.0, 1.0])
    
    state = jnp.stack([pos, vel], axis=1)  # shape (2, 2, 3)
    # state, mass = two_body_state()
    
    acc = force_func(state, mass, config, params, return_potential=False)

    # Analytical acceleration between two masses at 1 unit distance
    expected_force = 1.0  # G * m1 * m2 / r^2, with G=1
    expected_acc = jnp.array([
        [+1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])  # Force direction: along x-axis
    
    assert jnp.allclose(acc, expected_acc, atol=1e-5)
