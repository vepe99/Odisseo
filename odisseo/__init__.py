from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from odisseo.initial_condition import ic_two_body


"""
Odisseo package.

:mod:`odisseo` is a differentiable direct N-body simulation package written in JAX.

.. moduleauthor:: Giuseppe Viterbo (@vepe99)
.. :no-index:
"""

def construct_initial_state(position, velocity):
    """Constructs the initial state for the simulation.

    Args:
        position (jnp.ndarray): Initial positions of the bodies.
        velocity (jnp.ndarray): Initial velocities of the bodies.

    Returns:
        jnp.ndarray: The initial state containing positions and velocities.
    """
    state = jnp.zeros((position.shape[0], 2, position.shape[1]))
    state = state.at[:, 0, :].set(position)
    state = state.at[:, 1, :].set(velocity)
    
    return state

    
