from typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped, Array, Float, Scalar
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from odisseo.potentials import combined_external_acceleration, combined_external_acceleration_vmpa_switch
from odisseo.dynamics import direct_acc, direct_acc_laxmap, direct_acc_matrix, direct_acc_for_loop, direct_acc_sharding

from odisseo.option_classes import DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_SHARDING
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import DOPRI5, TSIT5, SEMIIMPLICITEULER, REVERSIBLEHEUN, LEAPFROGMIDPOINT

from diffrax import diffeqsolve, ODETerm, SaveAt
from diffrax import Tsit5, Dopri5
from diffrax import SemiImplicitEuler, ReversibleHeun, LeapfrogMidpoint


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def leapfrog(state: jnp.ndarray,
             mass: jnp.ndarray,
             dt: Scalar,
             config: SimulationConfig,
             params: SimulationParams):
    """
    Simple implementation of a symplectic Leapfrog (Verlet) integrator for N-body simulations.

    Args:
        state (jax.numpy.ndarray): The state of the particles, where the first column represents positions and the second column represents velocities.
        mass (jax.numpy.ndarray): The mass of the particles.
        dt (float): Time-step for current integration.
        config (object): Configuration object containing the acceleration scheme and external accelerations.
        params (dict): Additional parameters for the acceleration functions.
    Returns:
        jax.numpy.ndarray: The updated state of the particles.
    """
    if config.acceleration_scheme == DIRECT_ACC:
        acc_func = direct_acc
    
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        acc_func = direct_acc_laxmap

    elif config.acceleration_scheme == DIRECT_ACC_MATRIX:
        acc_func = direct_acc_matrix
    
    elif config.acceleration_scheme == DIRECT_ACC_FOR_LOOP:
        acc_func = direct_acc_for_loop
    
    elif config.acceleration_scheme == DIRECT_ACC_SHARDING:
        acc_func = direct_acc_sharding

    add_external_acceleration = len(config.external_accelerations) > 0
    
    acc = acc_func(state, mass, config, params)

    # Check additional accelerations
    if add_external_acceleration:
        acc = acc + combined_external_acceleration_vmpa_switch(state, config, params)
            
    # removing half-step velocity
    state = state.at[:, 0].set(state[:, 0] + state[:, 1]*dt + 0.5*acc*(dt**2))

    acc2 = acc_func(state, mass, config, params)

    if add_external_acceleration:
        acc2 = acc2 + combined_external_acceleration_vmpa_switch(state, config, params)
         
    state = state.at[:, 1].set(state[:, 1] + 0.5*(acc + acc2)*dt)
    
    return state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def RungeKutta4(state: jnp.ndarray,
             mass: jnp.ndarray,
             dt: Scalar,
             config: SimulationConfig,
             params: SimulationParams):
    """
    Simple implementation of a 4th order Runge-Kutta integrator for N-body simulations.

    Args:
        state (jax.numpy.ndarray): The state of the particles, where the first column represents positions and the second column represents velocities.
        mass (jax.numpy.ndarray): The mass of the particles.
        dt (float): Time-step for current integration.
        config (object): Configuration object containing the acceleration scheme and external accelerations.
        params (dict): Additional parameters for the acceleration functions.
    Returns:
        jax.numpy.ndarray: The updated state of the particles.
    """
    if config.acceleration_scheme == DIRECT_ACC:
        acc_func = direct_acc
    
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        acc_func = direct_acc_laxmap

    elif config.acceleration_scheme == DIRECT_ACC_MATRIX:
        acc_func = direct_acc_matrix

    add_external_acceleration = len(config.external_accelerations) > 0

    k1r = state[:, 1] * dt
    k1v = acc_func(state, mass, config, params) * dt

    state_2 = state.copy()
    state_2 = state_2.at[:, 0].set(state[:, 0] + 0.5*k1r)
    acc2 = acc_func(state_2, mass, config, params)
    if add_external_acceleration:
        acc2 = acc2 + combined_external_acceleration_vmpa_switch(state, config, params)

    k2r = (state[:, 1] + 0.5*k1v) * dt
    k2v = acc2 * dt

    state_3 = state.copy()
    state_3 = state_3.at[:, 0].set(state[:, 0] + 0.5*k2r)
    acc3 = acc_func(state_3, mass, config, params)
    if add_external_acceleration:
        acc3 = acc3 + combined_external_acceleration_vmpa_switch(state, config, params)

    k3r = (state[:, 1] + 0.5*k2v) * dt
    k3v = acc3 * dt

    state_4 = state.copy()
    state_4 = state_4.at[:, 0].set(state[:, 0] + k3r)
    acc4 = acc_func(state_4, mass, config, params)
    if add_external_acceleration:
        acc4 = acc4 + combined_external_acceleration_vmpa_switch(state, config, params)

    k4r = (state[:, 1] + k3v) * dt
    k4v = acc4 * dt

    state = state.at[:, 0].set(state[:, 0] + (k1r + 2*k2r + 2*k3r + k4r)/6)
    state = state.at[:, 1].set(state[:, 1] + (k1v + 2*k2v + 2*k3v + k4v)/6)

    return state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def diffrax_solver(state: jnp.ndarray,
                    mass: jnp.ndarray,
                    dt: Scalar,
                    config: SimulationConfig,
                    params: SimulationParams,) -> jnp.ndarray:
    """
    Diffrax backhand

    Args:
        state (jax.numpy.ndarray): The state of the particles, where the first column represents positions and the second column represents velocities.
        mass (jax.numpy.ndarray): The mass of the particles.
        dt (float): Time-step for current integration.
        config (object): Configuration object containing the acceleration scheme and external accelerations.
        params (dict): Additional parameters for the acceleration functions.

    Returns:
        jax.numpy.ndarray: The updated state of the particles.
     """

    def vector_field(t, y, args):
        """
        Vector field function for the ODE solver.

        Args:
            t (float): Time variable.
            y (jax.numpy.ndarray): State vector.
            args (tuple): Additional arguments for the acceleration function.

        Returns:
            jax.numpy.ndarray: The updated state of the particles.
        """
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = y
        # Unpack the args
        mass  = args

        positions = jnp.stack((pos_x, pos_y, pos_z), axis=1)
        velocities = jnp.stack((vel_x, vel_y, vel_z), axis=1)
        state = jnp.stack((positions, velocities), axis=1)

        d_xpos = vel_x
        d_ypos = vel_y
        d_zpos = vel_z

        d_vx = acc_func(state, mass, config, params)[:, 0] +  external_acc_func(state, config, params)[:, 0]
        d_vy = acc_func(state, mass, config, params)[:, 1] +  external_acc_func(state, config, params)[:, 1]
        d_vz = acc_func(state, mass, config, params)[:, 2] +  external_acc_func(state, config, params)[:, 2]

        d_y = jnp.array([d_xpos, d_ypos, d_zpos, d_vx, d_vy, d_vz])

        return d_y

    
    def f(t, y, args):
        """
        Vector field for the transform of positions
        """
        return y
    
    def g(t, y, args):
        """
        Vector field for the transform of velocities
        args is the mass
        """
        state = jnp.zeros((config.N_particles, 2, 3))
        state = state.at[:, 0].set(y)

        return acc_func(state, args, config, params) + external_acc_func(state, config, params)


    if config.acceleration_scheme == DIRECT_ACC:
        acc_func = direct_acc
    
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        acc_func = direct_acc_laxmap

    elif config.acceleration_scheme == DIRECT_ACC_MATRIX:
        acc_func = direct_acc_matrix

    add_external_acceleration = len(config.external_accelerations) > 0

    if add_external_acceleration:
        external_acc_func = combined_external_acceleration_vmpa_switch
    else:
        external_acc_func = lambda state, config, params: jnp.zeros_like(state[:, 0])

    if config.diffrax_solver == DOPRI5:
        solver = Dopri5()
        term = ODETerm(vector_field)
    elif config.diffrax_solver == TSIT5:
        solver = Tsit5()
        term = ODETerm(vector_field)

    # Symplectic methods
    elif config.diffrax_solver == SEMIIMPLICITEULER:
        solver = SemiImplicitEuler()
        term = (ODETerm(f), ODETerm(g))
    elif config.diffrax_solver == REVERSIBLEHEUN:
        solver = ReversibleHeun()
        term = ODETerm(vector_field)
    elif config.diffrax_solver == LEAPFROGMIDPOINT:
        solver = LeapfrogMidpoint()
        term = ODETerm(vector_field)
    
    if config.diffrax_solver != SEMIIMPLICITEULER:
        t0 = 0.0
        dt0 = dt
        t1 = dt #in the fixed number of timesteps case we want to integrate only one step
        y0 = jnp.array([state[:, 0, 0], state[:, 0, 1], state[:, 0, 2], state[:, 1, 0], state[:, 1, 1], state[:, 1, 2]])
        args = mass
        sol = diffeqsolve(
            terms = term,
            solver = solver,
            t0 = t0,
            t1 = t1,
            dt0 = dt0,
            y0 = y0,
            args=args,)
        pos = jnp.stack((sol.ys[0][0], sol.ys[0][1], sol.ys[0][2]), axis=1)
        vel = jnp.stack((sol.ys[0][3], sol.ys[0][4], sol.ys[0][5]), axis=1)

    else:
        t0 = 0.0
        dt0 = dt
        t1 = dt #in the fixed number of timesteps case we want to integrate only one step
        y0 = jnp.array([state[:, 0, 0], state[:, 0, 1], state[:, 0, 2]]), jnp.array([state[:, 1, 0], state[:, 1, 1], state[:, 1, 2]])
        args = mass
        sol = diffeqsolve(
            terms = term,
            solver = solver,
            t0 = t0,
            t1 = t1,
            dt0 = dt0,
            y0 = y0,
            args=args,)
        pos = jnp.stack((sol.ys[0][0], sol.ys[0][1], sol.ys[0][2]), axis=1)
        vel = jnp.stack((sol.ys[1][0], sol.ys[1][1], sol.ys[1][2]), axis=1)

    return jnp.stack((pos, vel), axis=1) 

 

     

