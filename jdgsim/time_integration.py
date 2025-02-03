from timeit import default_timer as timer
from functools import partial
from typing import Union

import jax
from jax import jit
from jaxtyping import Array, Float, jaxtyped

from jdgsim.integrators import leapfrog
from jdgsim.option_classes import SimulationConfig, SimulationParams
from jdgsim.integrators import LEAPFROG, leapfrog


@partial(jax.jit, static_argnames=['config',])
def time_integration(primitive_state, mass, config: SimulationConfig, params: SimulationParams, ):
    """Integrate the fluid equations in time. For the options of
    the time integration see the simulation configuration and
    the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution.

    """

    if config.fixed_timestep:
        return _time_integration_fixed_steps(primitive_state, mass, config, params)
    else:
        raise NotImplementedError("Adaptive time stepping not implemented yet")
        # if config.differentiation_mode == BACKWARDS:
        #     return _time_integration_adaptive_backwards(primitive_state, config, params, helper_data, registered_variables)
        # else:
        #     return _time_integration_adaptive_steps(primitive_state, config, params, helper_data, registered_variables)
        

@partial(jax.jit, static_argnames=['config'])
def _time_integration_fixed_steps(primitive_state, mass, config: SimulationConfig, params: SimulationParams, ):
    """ Fixed time stepping integration of the fluid equations.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution
    """

    if config.return_snapshots:
        raise NotImplementedError("not impelemented yet")

    dt = params.t_end / config.num_timesteps
    
    def update_step(_, state):
        if config.integrator == LEAPFROG:
            return leapfrog(state, mass, dt, config, params)

    # use lax fori_loop to unroll the loop
    state = jax.lax.fori_loop(0, config.num_timesteps, update_step, primitive_state)

    return state  

    
    