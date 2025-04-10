from typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
from jax import random
from jax.scipy.special import gammaln, gammainc


from astropy import units as u

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import NFW_POTENTIAL, POINT_MASS, MN_POTENTIAL, PSP_POTENTIAL
from odisseo.units import CodeUnits

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential',])
def combined_external_acceleration(state: jnp.ndarray, 
                                   config: SimulationConfig,
                                   params: SimulationParams,
                                   return_potential=False):
    """
    Compute the total acceleration of all particles due to all external potentials. Sequential way 
    
    Args:
        state (jnp.ndarray): Array of shape (N_particles,2,3) representing the positions and velocities of the particles. 
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool): If True, also returns the total potential energy of all external potentials.
    
    Returns:
        jnp.ndarray: Total acceleration of all particles due to all external potentials if return_potential is False.
        Tuple: Total acceleration and total potential energy of all particles due to all external potentials if return_potential is True.
    """
    total_external_acceleration = jnp.zeros_like(state[:, 0])
    total_external_potential = jnp.zeros_like(config.N_particles)
    if return_potential:
        if NFW_POTENTIAL in config.external_accelerations:
            acc_NFW, pot_NFW = NFW(state, config, params, return_potential=True)
            total_external_acceleration = total_external_acceleration + acc_NFW
            total_external_potential = total_external_potential +   pot_NFW
            if POINT_MASS in config.external_accelerations:
                acc_PM, pot_PM = point_mass(state, config, params, return_potential=True)
                total_external_acceleration = total_external_acceleration + acc_PM
                total_external_potential = total_external_potential + pot_PM
                if MN_POTENTIAL in config.external_accelerations:
                    acc_MN, pot_MN = MyamotoNagai(state, config, params, return_potential=True)
                    total_external_acceleration = total_external_acceleration + acc_MN
                    total_external_potential = total_external_potential + pot_MN
                    return total_external_acceleration, total_external_potential
                else:
                    return total_external_acceleration, total_external_potential
        return total_external_acceleration, total_external_potential
    else:
        if NFW_POTENTIAL in config.external_accelerations:
            total_external_acceleration = total_external_acceleration + NFW(state, config, params)
        return total_external_acceleration

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential', 'code_units'])    
def combined_external_acceleration_vmpa_switch(state: jnp.ndarray, 
                                                config: SimulationConfig,
                                                params: SimulationParams,
                                                code_units: CodeUnits,
                                                return_potential=False):

    """
    Compute the total acceleration of all particles due to all external potentials.
    Vectorized way

    Args:
        state (jnp.ndarray): Array of shape (N_particles,2,3) representing the positions and velocities of the particles. 
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool): If True, also returns the total potential energy of all external potentials.
    
    Returns:
        jnp.ndarray: Total acceleration of all particles due to all external potentials if return_potential is False.
        Tuple: Total acceleration and total potential energy of all particles due to all external potentials if return_potential is True.

    """

    total_external_acceleration = jnp.zeros_like(state[:, 0])
    total_external_potential = jnp.zeros_like(config.N_particles)
    state_tobe_vmap  = jnp.repeat(state[jnp.newaxis, ...], repeats=len(config.external_accelerations), axis=0)
    if return_potential:
        # The POTENTIAL_LIST NEEDS TO BE IN THE SAME ORDER AS THE INTEGER VALUES 
        POTENTIAL_LIST = [lambda state: NFW(state, config=config, params=params, code_units=code_units, return_potential=True), 
                        lambda state: point_mass(state, config=config, params=params, code_units=code_units, return_potential=True),
                        lambda state: MyamotoNagai(state, config=config, params=params, code_units=code_units, return_potential=True),
                        lambda state: PowerSphericalPotentialwCutoff(state, config=config, params=params, code_units=code_units, return_potential=True)]
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc, external_pot = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        total_external_potential = jnp.sum(external_pot, axis=0)
        return total_external_acceleration, total_external_potential
    else:
        POTENTIAL_LIST = [lambda state: NFW(state, config=config, params=params, code_units=code_units, return_potential=False),
                          lambda state: point_mass(state, config=config, params=params, code_units=code_units, return_potential=False),
                          lambda state: MyamotoNagai(state, config=config, params=params, code_units=code_units, return_potential=False),
                          lambda state: PowerSphericalPotentialwCutoff(state, config=config, code_units=code_units, params=params, return_potential=False)]
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        return total_external_acceleration

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential', 'code_units'])
def NFW(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
        code_units: CodeUnits,
        return_potential=False,):
    """
    Compute acceleration of all particles due to a NFW profile.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 2, 3) representing the positions and velocities of the particles.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool, optional): If True, also returns the potential energy of the NFW profile. Defaults to False.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Acceleration (jnp.ndarray): Acceleration of all particles due to NFW external potential.
            - Potential (jnp.ndarray): Potential energy of all particles due to NFW external potential. Returned only if return_potential is True.
    """
    
    params_NFW = params.NFW_params
    
    r  = jnp.linalg.norm(state[:, 0], axis=1)

    NUM = (params_NFW.r_s+r)*jnp.log(1+r/params_NFW.r_s) - r
    DEN = r*r*r*(params_NFW.r_s+r)*params_NFW.d_c

    fNFW = 0.35
    R0kpc = 8.0 * u.kpc.to(code_units.code_length)
    vc0kms = 220.0 * (u.km / u.s).to(code_units.code_length/code_units.code_time)

    ftot = vc0kms**2/R0kpc

    @jit
    def acceleration(state):
        # return - params.G * params_NFW.Mvir*NUM[:, jnp.newaxis]/DEN[:, jnp.newaxis] * state[:, 0]
        rad = r
        dimless_prefactor = (
            R0kpc**2 * (rad / (params_NFW.r_s + rad) - jnp.log((params_NFW.r_s + rad) / params_NFW.r_s))
        ) / (
            rad**2
            * (R0kpc / (params_NFW.r_s + R0kpc) - jnp.log((params_NFW.r_s + R0kpc) / params_NFW.r_s))
        )
        direction = (1 / rad[:, None]) * state[:, 0]
        return -fNFW * ftot * dimless_prefactor[:, None] * direction


    @jit 
    def potential(state):
        return - params.G * params_NFW.Mvir*jnp.log(1+r/params_NFW.r_s)/(r*params_NFW.d_c)
    
    acc = acceleration(state)

    if return_potential:
        pot = potential(state)
        return acc, pot
    else:
        return acc
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential', 'code_units'])
def point_mass(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
        code_units: CodeUnits,
        return_potential=False):
    """
    Compute acceleration of all particles due to a point mass potential.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 2, 3) representing the positions and velocities of the particles.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool, optional): If True, also returns the potential energy of the point mass potential. Defaults to False.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Acceleration (jnp.ndarray): Acceleration of all particles due to point mass external potential.
            - Potential (jnp.ndarray): Potential energy of all particles due to point mass external potential. Returned only if return_potential is True.
    """
    params_point_mass = params.PointMass_params
    
    r  = jnp.linalg.norm(state[:, 0], axis=1)

    @jit
    def acceleration(state):
        return - params.G * params_point_mass.M * state[:, 0] / (r**3)[:, None]
    
    @jit
    def potential(state):
        return - params.G * params_point_mass.M / r
    
    acc = acceleration(state)
    
    if return_potential:
        pot = potential(state)
        return acc, pot
    else:
        return acc
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential', 'code_units'])
def MyamotoNagai(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
        code_units: CodeUnits,
        return_potential=False):
    """
    Compute acceleration of all particles due to a MyamotoNagai disk profile.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 2, 3) representing the positions and velocities of the particles.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool, optional): If True, also returns the potential energy of the MyamotoNagai profile. Defaults to False.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Acceleration (jnp.ndarray): Acceleration of all particles due to MyamotoNagai external potential.
            - Potential (jnp.ndarray): Potential energy of all particles due to MyamotoNagai external potential. Returned only if return_potential is True.
    """
    params_MN = params.MN_params
    b = params_MN.b
    a = params_MN.a

    fD = 0.6
    R0kpc = 8.0 * u.kpc.to(code_units.code_length)
    vc0kms = 220.0 * (u.km / u.s).to(code_units.code_length/code_units.code_time)

    ftot = vc0kms**2/R0kpc

    z2 = state[:, 0, 2]**2
    
    Dz = (a+(z2+b**2)**0.5)
    D = jnp.sum(state[:, 0, :2]**2, axis=1) + Dz**2
    K = params.G * params_MN.M / D**(3/2)

    # @jit
    # def acceleration(state):
    #     ax = -K * state[:, 0, 0]
    #     ay = -K * state[:, 0, 1]
    #     az = -K * state[:, 0, 2] * Dz / (z2 + b**2)**0.5
    #     return jnp.stack([ax, ay, az], axis=1)

    @jit
    def acceleration(state):
        R2 = state[:, 0, 0]**2 + state[:, 0, 1]**2
        dimless_prefactor = (
            (R0kpc**2 + (a + b) ** 2)
            / (R2 + (a + jnp.sqrt(b**2 + state[:, 0, 2] ** 2)) ** 2)
        ) ** (3 / 2)
        # direction = (1 / R0kpc) * jnp.array(
        #     [
        #         state[:, 0, 0],
        #         state[:, 0, 1],
        #         state[:, 0, 2]
        #         * (a + jnp.sqrt(b**2 + state[:, 0, 2] ** 2))
        #         / jnp.sqrt(b**2 + state[:, 0, 2] ** 2),
        #     ]
        # )
        direction = (1 / R0kpc) * state[:, 0]
        direction = direction.at[:, 2].set( 
            state[:, 0, 2] * (a + jnp.sqrt(b**2 + state[:, 0, 2] ** 2)) / jnp.sqrt(b**2 + state[:, 0, 2] ** 2)
        )
        return -fD * ftot * dimless_prefactor[:, None] * direction

    
    @jit
    def potential(state):
        return - params.G * params_MN.M / jnp.sqrt(D)

    
    acc = acceleration(state)

    if return_potential:
        pot = potential(state)
        return acc, pot
    else:
        return acc
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential', 'code_units'])
def PowerSphericalPotentialwCutoff(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
        code_units: CodeUnits,
        return_potential=False):
    """
    Compute acceleration of all particles due to a power spherical potential with cutoff.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 2, 3) representing the positions and velocities of the particles.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool, optional): If True, also returns the potential energy of the power spherical potential. Defaults to False.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Acceleration (jnp.ndarray): Acceleration of all particles due to power spherical external potential.
            - Potential (jnp.ndarray): Potential energy of all particles due to power spherical external potential. Returned only if return_potential is True.
    """
    params_PSP = params.PSP_params
    alpha = params_PSP.alpha
    r_c = params_PSP.r_c
    fB = 0.05
    R0kpc = 8.0 * u.kpc.to(code_units.code_length)
    vc0kms = 220.0 * (u.km / u.s).to(code_units.code_length/code_units.code_time)

    ftot = vc0kms**2/R0kpc
    
    r = jnp.linalg.norm(state[:, 0], axis=1)

    @jit
    def rho(radius):
        return (1/radius)**-alpha * jnp.exp(-(radius/r_c)**2) 

    @jit
    def enclosed_mass(radius):
        #integration for the enclosed mass
        r_1 = jnp.linspace(0, radius, 1000)
        rho_1 = rho(r_1)
        return 4 * jnp.pi * jax.scipy.integrate.trapezoid(rho_1*r_1**2, r_1)

    M_enc = vmap(enclosed_mass)(r)

    @jax.jit
    def gamma(x: float) -> float:
        """
        Compiled version of the standard Gamma function
        Args:
        x: input value
        Returns:
        Gamma function evaludated at x
        Examples
        --------
        >>> gamma(2.)
        """
        return jnp.exp(gammaln(x))
    
    g = gamma(1.5 - (alpha / 2))


    @jax.jit
    def gamma_low(x: float, y: float) -> float:
        """
        Compiled version of the incomplete gamma function from below (integral from 0 to y)
        Args:
        x: input value
        y: upper integration limit
        Returns:
        Incomplete gamma function from below evaludated at x
        Examples
        --------
        >>> gamma_low(2., 10.)
        """
        return jnp.exp(gammaln(x)) * (1.0 - gammainc(x, y))

    # @jit
    # def acceleration(state):
    #     return - params.G * state[:, 0] / (r**3)[:, None]

    @jit
    def acceleration(state):
        rad = r
        dimless_prefactor = (
            R0kpc**2
            * (g - gamma_low(1.5 - (alpha / 2), (rad / r_c) ** 2))
        ) / (
            rad**2
            * (g - gamma_low(1.5 - (alpha / 2), (R0kpc / r_c) ** 2))
        )
        direction = (1 / rad[:, None]) * state[:, 0]
        return -fB * ftot * dimless_prefactor[:, None] * direction


    @jit 
    def potential(state):
        return - params.G * M_enc / r

    acc = acceleration(state)
    
    if return_potential:
        pot = potential(state)
        return acc, pot
    else:
        return acc