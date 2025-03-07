from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random





def construct_initial_state(position, velocity):
    state = jnp.zeros((position.shape[0], 2, position.shape[1]))
    state = state.at[:, 0, :].set(position)
    state = state.at[:, 1, :].set(velocity)
    
    return state

    
