from timeit import default_timer as timer
from functools import partial
from typing import Union, NamedTuple
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import jax
from jax import jit
import jax.numpy as jnp


from equinox.internal._loop.checkpointed import checkpointed_while_loop
import equinox as eqx


from odisseo.integrators import leapfrog
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import LEAPFROG, RK4
from odisseo.integrators import leapfrog,RungeKutta4
from odisseo.utils import E_tot, Angular_momentum

class SnapshotData(NamedTuple):
    """Return format for the time integration, when snapshots are requested."""

    #: The times at which the snapshots were taken.
    times: jnp.ndarray = None

    #: The primitive states at the times the snapshots were taken.
    states: jnp.ndarray = None

    #: The total energy at the times the snapshots were taken.
    total_energy: jnp.ndarray = None
    
    #: The angular momentum at the times the snapshots were taken.
    angular_momentum: jnp.ndarray = None

    # The runtime of the simulation-loop.
    runtime: float = 0.0

    #: Number of timesteps taken.
    num_iterations: int = 0

    #: The current checkpoint, used internally.
    current_checkpoint: int = 0


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config',])
def time_integration(primitive_state: jnp.ndarray,
                     mass: jnp.ndarray,
                     config: SimulationConfig,
                     params: SimulationParams, ):
    """
    Integrate the Nbody simulation in time. For the options of
    the time integration see the simulation configuration and
    the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution
    """

    if config.fixed_timestep:
        if config.return_snapshots:
            return _time_integration_fixed_steps_snapshot(primitive_state, mass, config, params)
        else:
            return _time_integration_fixed_steps(primitive_state, mass, config, params)
    
    else:
        raise NotImplementedError("Adaptive time stepping not implemented yet")

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _time_integration_fixed_steps(primitive_state: jnp.ndarray,
                                mass: jnp.ndarray,
                                config: SimulationConfig,
                                params: SimulationParams, ):

    """ Fixed time stepping integration of the primitave state of the system.
    Return the final state of the system after the time integration.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution
    """

    dt = params.t_end / config.num_timesteps
    
    def update_step(_, state):
        
        if config.integrator == LEAPFROG:
            return leapfrog(state, mass, dt, config, params)
        elif config.integrator == RK4:
            return RungeKutta4(state, mass, dt, config, params)
            

    # use lax fori_loop to unroll the loop
    state = jax.lax.fori_loop(0, config.num_timesteps, update_step, primitive_state)

    return state  

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _time_integration_fixed_steps_snapshot(primitive_state: jnp.ndarray,
                     mass: jnp.ndarray,
                     config: SimulationConfig,
                     params: SimulationParams, ):  
    """ Fixed time stepping integration of the primitave state of the system.
    Return the snapshot of the state of the system at fixed point in the time integration.

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
        times = jnp.zeros(config.num_snapshots)
        states = jnp.zeros((config.num_snapshots, primitive_state.shape[0], primitive_state.shape[1], primitive_state.shape[2]))
        total_energy = jnp.zeros(config.num_snapshots)
        angular_momentum = jnp.zeros((config.num_snapshots, 3))
        current_checkpoint = 0
        snapshot_data = SnapshotData(times = times, 
                                     states = states, 
                                     total_energy = total_energy, 
                                     angular_momentum = angular_momentum,
                                     current_checkpoint = current_checkpoint)

    def update_step(carry):

        if config.return_snapshots:
            time, state, snapshot_data = carry

            def update_snapshot_data(snapshot_data):
                times = snapshot_data.times.at[snapshot_data.current_checkpoint].set(time)
                states = snapshot_data.states.at[snapshot_data.current_checkpoint].set(state)
                total_energy = snapshot_data.total_energy.at[snapshot_data.current_checkpoint].set(E_tot(state, mass, config, params))
                angular_momentum = snapshot_data.angular_momentum.at[snapshot_data.current_checkpoint].set(Angular_momentum(state, mass))
                current_checkpoint = snapshot_data.current_checkpoint + 1
                snapshot_data = snapshot_data._replace(times = times, 
                                                       states = states, 
                                                       total_energy = total_energy, 
                                                       angular_momentum = angular_momentum,
                                                       current_checkpoint = current_checkpoint)
                return snapshot_data
            
            def dont_update_snapshot_data(snapshot_data):
                return snapshot_data

            snapshot_data = jax.lax.cond(time >= snapshot_data.current_checkpoint * params.t_end / config.num_snapshots, update_snapshot_data, dont_update_snapshot_data, snapshot_data)

            num_iterations = snapshot_data.num_iterations + 1
            snapshot_data = snapshot_data._replace(num_iterations = num_iterations)

        else:
            time, state = carry

        dt = params.t_end / config.num_timesteps
        
        if config.integrator == LEAPFROG:
            state = leapfrog(state, mass, dt, config, params)
        elif config.integrator == RK4:
            state = RungeKutta4(state, mass, dt, config, params)

        time += dt

        if config.return_snapshots:
            carry = (time, state, snapshot_data)
        else:
            carry = (time, state)

        return carry
    
    def condition(carry):
        if config.return_snapshots:
            t, _, _ = carry
        else:
            t, _ = carry
        return t < params.t_end
    
    if config.return_snapshots:
        carry = (0.0, primitive_state, snapshot_data)
    else:
        carry = (0.0, primitive_state)
    
    start = timer()
    carry = jax.lax.while_loop(condition, update_step, carry)
    # carry = checkpointed_while_loop(condition, update_step, carry)
    # carry = jax.lax.fori_loop(0, config.num_timesteps, update_step, carry)
    end = timer()
    duration = end - start

    if config.return_snapshots:
        _, state, snapshot_data = carry
        snapshot_data = snapshot_data._replace(runtime = duration)
        return snapshot_data
    else:
        _, state = carry
        return state
    