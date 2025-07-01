from typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
from jax import random

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import NFW_POTENTIAL, POINT_MASS, MN_POTENTIAL, PSP_POTENTIAL

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
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
@partial(jax.jit, static_argnames=['config', 'return_potential'])    
def combined_external_acceleration_vmpa_switch(state: jnp.ndarray, 
                                                config: SimulationConfig,
                                                params: SimulationParams,
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
        POTENTIAL_LIST = [lambda state: NFW(state, config=config, params=params, return_potential=True), 
                          lambda state: point_mass(state, config=config, params=params, return_potential=True),
                          lambda state: MyamotoNagai(state, config=config, params=params, return_potential=True),
                          lambda state: PowerSphericalPotentialwCutoff(state, config=config, params=params, return_potential=True), 
                          lambda state: logarithmic_potential(state, config=config, params=params, return_potential=True)]
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc, external_pot = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        total_external_potential = jnp.sum(external_pot, axis=0)
        return total_external_acceleration, total_external_potential
    else:
        POTENTIAL_LIST = [lambda state: NFW(state, config=config, params=params, return_potential=False),
                          lambda state: point_mass(state, config=config, params=params, return_potential=False),
                          lambda state: MyamotoNagai(state, config=config, params=params, return_potential=False),
                          lambda state: PowerSphericalPotentialwCutoff(state, config=config, params=params, return_potential=False),
                          lambda state: logarithmic_potential(state, config=config, params=params, return_potential=False)]
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        return total_external_acceleration

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def NFW(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
        return_potential=False):
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
    
    # params_NFW = params.NFW_params
    
    # r  = jnp.linalg.norm(state[:, 0], axis=1)

    # NUM = (params_NFW.r_s+r)*jnp.log(1+r/params_NFW.r_s) - r
    # DEN = r*r*r*(params_NFW.r_s+r)*params_NFW.d_c

    # @jit
    # def acceleration(state):
    #     return - params.G * params_NFW.Mvir*NUM[:, jnp.newaxis]/DEN[:, jnp.newaxis] * state[:, 0]

    
    # @jit 
    # def potential(state):
    #     return - params.G * params_NFW.Mvir*jnp.log(1+r/params_NFW.r_s)/(r*params_NFW.d_c)
    
    # acc = acceleration(state)

    # if return_potential:
    #     pot = potential(state)
    #     return acc, pot
    # else:
    #     return acc

    params_NFW = params.NFW_params
    M = params_NFW.Mvir
    r_s = params_NFW.r_s

    r = jnp.linalg.norm(state[:, 0], axis=1)

    @jit
    def potential(r):
        r"""Potential for the NFW model.

        $$ \Phi(r) = -\frac{G m}{r_s} \frac{r_s}{r} \log(1 + \frac{r}{r_s}) $$

        where $m$ is the characteristic mass and $r_s$ is the scale radius.

        """
        x = r / r_s
        phi0 = -params.G * M / r_s
        return phi0 * jnp.log(1 + x) / x
    
    @jit
    def mass_enclosed(r):
        r"""Enclosed mass for the NFW model.

        $$ M(<r) = \frac{m}{\ln(1 + x) - \frac{x}{1 + x}} $$

        where $x = r / r_s$ is the dimensionless radius and $m$ is the
        characteristic mass.

        """
        x = r / r_s
        return M * (jnp.log(1 + x) - x / (1 + x))
    
    @jit 
    def acceleration(r):
        return - params.G * mass_enclosed(r)[:, None] * state[:, 0] / (r**3)[:, None] 
    
    # @jit
    # def acceleration(r):
    #     rad = jnp.linalg.norm(state[:, 0], axis=1)
    #     dimless_prefactor = (
    #         8**2 * (rad / (r_s + rad) - jnp.log((r_s + rad)/r_s) 
    #         / (rad**2 * (8./ (r_s + 8.)) - jnp.log((r_s+8.)/r_s)) 
    #     ))
    #     direction = (1/rad)[:, None ] * state[:, 0]
    #     ftot = (0.000001045940172532453 * 220**2 / 8.) * 1
    #     return - 0.35 * ftot * dimless_prefactor[:, None] * direction

    #calculate the acceleration
    acc = acceleration(r)

    if return_potential:
        pot = potential(r)
        return acc, pot
    else:
        return acc
    
    


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def point_mass(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
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
    def potential(r):
        return - params.G * params_point_mass.M / r
    
    acc = acceleration(state)
    
    if return_potential:
        pot = potential(r)
        return acc, pot
    else:
        return acc
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def MyamotoNagai(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
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
    
    z2 = state[:, 0, 2]**2
    b = params_MN.b
    a = params_MN.a
    M = params_MN.M

    Dz = (a+(z2+b**2)**0.5)
    D = jnp.linalg.norm(state[:, 0, :2], axis=1)**2 + Dz**2
    K = - params.G * params_MN.M / D**(3/2)

    @jit
    def acceleration(pos):
        ax = K * pos[:, 0]
        ay = K * pos[:, 1]
        az = K * pos[:, 2] * Dz / (z2 + b**2)**0.5
        return jnp.stack([ax, ay, az], axis=1)

    @jit
    def potential(pos):
        return - params.G * params_MN.M / jnp.sqrt(D)

    # def acceleration(state):
    #     R2 = state[:, 0, 0]**2 + state[:, 0, 1]**2
    #     dimless_prefactor = ((8.**2 + (a + b)**2) / (R2 + (a + jnp.sqrt(b**2 + state[:, 0, 2]**2))**2 ))**(3/2)
    #     direction = (1 / 8.) * jnp.array([
    #         state[:, 0, 0],
    #         state[:, 0, 1],
    #         state[:, 0, 2] * (a + jnp.sqrt(b**2 + state[:, 0, 2]**2))/jnp.sqrt(b**2 + state[:, 0, 2]**2)
    #     ]).T

    #     ftot = (0.000001045940172532453 * 220**2 / 8.) * 1

    #     return  - 0.6  * ftot * dimless_prefactor[:, None] * direction

    

    # @jit
    # def potential(pos):
    #     R2 = jnp.linalg.norm(pos[:2])**2
    #     zp2 = (jnp.sqrt(pos[2]**2 + b**2) +a )**2
    #     return -params.G * M / jnp.sqrt(R2 + zp2)
    
    # @jit 
    # def acceleration(pos):
    #     return -jax.vmap(jax.grad((potential)))(pos)

    pos = state[:, 0]
    acc = acceleration(pos)

    if return_potential:
        pot = potential(pos)
        return acc, pot
    else:
        return acc
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def PowerSphericalPotentialwCutoff(state: jnp.ndarray, 
        config: SimulationConfig,
        params: SimulationParams,
        return_potential=False):
    """
    Compute acceleration of all particles due to a power spherical potential with cutoff.
    taken from galax: https://github.com/GalacticDynamics/galax/blob/main/src/galax/potential/_src/builtin/powerlawcutoff.py#L35
    
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

    @partial(jax.jit)
    def _safe_gamma_inc(a, x):
        return jax.scipy.special.gammainc(a, x) * jax.scipy.special.gamma(a)
    
    params_PSP = params.PSP_params
    M = params_PSP.M
    alpha = params_PSP.alpha
    r_c = params_PSP.r_c
    
    pos = state[:, 0]

    @jit
    def rho(radius):
        return (1/radius)**alpha * jnp.exp(-(radius/r_c)**2) 

    @jit
    def potential(pos):
        r = jnp.linalg.norm(pos)
        a = alpha/2
        s2 = (r/r_c)**2
        GM = params.G * M
        pot_value =  GM * ((a - 1.5) * _safe_gamma_inc(1.5 - 1, s2) / (r * jax.scipy.special.gamma(2.5 - a)) 
                    + _safe_gamma_inc(1 - a, s2) / (r_c * jax.scipy.special.gamma(1.5 - a)))   
        return jnp.squeeze(pot_value)
    
    @jit 
    def acceleration(pos):
        return -jax.vmap(jax.grad((potential)))(pos)
    
    # def _mass(R,):
    #     out = (
    #         2.0
    #         * jnp.pi
    #         * R ** (3.0 - alpha)
    #         / (1.5 - alpha / 2.0)
    #         * jax.scipy.special.hyp1f1(
    #             1.5 - alpha / 2.0,
    #             2.5 - alpha / 2.0,
    #             -((R / r_c) ** 2.0),
    #         )
    #     )
    #     return out
    
    # @jit
    # def acceleration(pos):
    #     r = jnp.linalg.norm(pos, axis=1)
         
    #     return - params.G * _mass(r)[:, None] * pos / (r**3)[:, None]
    

    # @jax.jit
    # def gamma_low(x: float, y: float) -> float:
    #     """
    #     Compiled version of the incomplete gamma function from below (integral from 0 to y)
    #     Args:
    #     x: input value
    #     y: upper integration limit
    #     Returns:
    #     Incomplete gamma function from below evaludated at x
    #     Examples
    #     --------
    #     >>> gamma_low(2., 10.)
    #     """
    #     return jnp.exp(jax.scipy.special.gammaln(x)) * (1.0 - jax.scipy.special.gammainc(x, y))

    # def acceleration(pos):
    #     rad = jnp.linalg.norm(pos, axis=1)
    #     g = jnp.exp(jax.scipy.special.gammaln(1.5 - (alpha / 2)))
    #     dimensionless_prefactor = (
    #         8.**2 
    #         * (g - gamma_low(1.5 - (alpha /2 ), (rad / r_c)**2) )
    #         ) / ( rad**2 * (g - gamma_low(1.5 - (alpha /2), (8.0/ r_c)**2))
    #     )
    #     direction = (1/rad)[:, None ] * state[:, 0]
    #     ftot = (0.000001045940172532453 * 220**2 / 8.) * 1
    #     return - 0.05 * ftot * dimensionless_prefactor[:, None] * direction

    # compute the acceleration

    acc = acceleration(pos)
    if return_potential:
        pot = -jax.vmap(potential)(pos)
        return acc, pot
    else:
        return acc
 
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])  
def logarithmic_potential(state: jnp.ndarray,
                          config: SimulationConfig,
                          params: SimulationParams,
                          return_potential=False):
    """
    Compute acceleration of all particles due to a logarithmic potential.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 2, 3) representing the positions and velocities of the particles.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool, optional): If True, also returns the potential energy of the logarithmic potential. Defaults to False.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Acceleration (jnp.ndarray): Acceleration of all particles due to logarithmic external potential.
            - Potential (jnp.ndarray): Potential energy of all particles due to logarithmic external potential. Returned only if return_potential is True.
    """
    r = jnp.sqrt(state[:, 0, 0]**2 + state[:, 0, 1]**2)
    z = state[:, 0, 2]
    v2_0 = params.Logarithmic_params.v0**2
    q2 = params.Logarithmic_params.q**2
    
    @jit
    def potential(state):
        return - v2_0/2 * jnp.log(r**2 + (z**2/q2))

    @jit
    def acceleration(state):
        DEN = r**2 + (z**2/q2)
        ax = - v2_0 * state[:, 0, 0] / DEN
        ay = - v2_0 * state[:, 0, 1] / DEN
        az = - v2_0 * z * (1/q2) / DEN
        return jnp.stack([ax, ay, az], axis=1)
    
    acc = acceleration(state)
    
    if return_potential:
        pot = potential(state)
        return acc, pot
    else:
        return acc