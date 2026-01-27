from typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
from jax import random
import jax.scipy.special as jsp
from scipy.special import j0, j1, jv
import numpy as np
import scipy

from odisseo.option_classes import SimulationConfig, SimulationParams


@partial(jax.jit, static_argnames=['config', 'return_potential'])   
@jaxtyped(typechecker=typechecker)
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
                          lambda state: logarithmic_potential(state, config=config, params=params, return_potential=True),
                          lambda state: TriaxialNFW(state, config=config, params=params, return_potential=True),
                        #   lambda state: Thin_DoubleExponentialDiskPotential(state, config=config, params=params, return_potential=True),
                        #   lambda state: Thick_DoubleExponentialDiskPotential(state, config=config, params=params, return_potential=True),
                          ]  
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
                          lambda state: logarithmic_potential(state, config=config, params=params, return_potential=False),
                          lambda state: TriaxialNFW(state, config=config, params=params, return_potential=False),
                        #   lambda state: Thin_DoubleExponentialDiskPotential(state, config=config, params=params, return_potential=False),
                        #   lambda state: Thick_DoubleExponentialDiskPotential(state, config=config, params=params, return_potential=False),
                          ]  
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        return total_external_acceleration

@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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
    
    


@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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
    
@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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
    

@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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
        # pot_value =  - GM * (
        #     (a - 1.5) * _safe_gamma_inc(1.5 - 1, s2) / (r * jax.scipy.special.gamma(2.5 - a)) 
        #             + _safe_gamma_inc(1 - a, s2) / (r_c * jax.scipy.special.gamma(1.5 - a)))   
        # return jnp.squeeze(pot_value)

        den = jsp.gamma(1.5 -a)
        L1 = _safe_gamma_inc(1.5 - a, s2)
        L2 = _safe_gamma_inc(1 - a, s2)
        pot = -GM / den * ( L1 / r + (jsp.gamma(1 - a) - L2) / r_c )
        return jnp.squeeze(pot)

    
    @jit 
    def acceleration(pos):
        return -jax.vmap(jax.grad((potential)))(pos)
    

    # compute the acceleration
    acc = acceleration(pos)
    if return_potential:
        pot = jax.vmap(potential)(pos)
        return acc, pot
    else:
        return acc
 
@partial(jax.jit, static_argnames=['config', 'return_potential'])  
@jaxtyped(typechecker=typechecker)
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
    

@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
def TriaxialNFW(state: jnp.ndarray, 
                config: SimulationConfig,
                params: SimulationParams,
                return_potential=False):
    """
    Compute acceleration of all particles due to a Triaxial NFW profile.
    This code is heavily inspired by the implementation in galax: https://github.com/GalacticDynamics/galax/blob/main/src/galax/potential/_src/builtin/nfw/triaxial.py
    
    The density is given by:
        rho(xi) = rho_0 / (xi/r_s) / (1 + xi/r_s)^2
    where:
        xi^2 = x^2 + y^2/q1^2 + z^2/q2^2

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 2, 3) representing the positions and velocities of the particles.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool, optional): If True, also returns the potential energy. Defaults to False.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Acceleration (jnp.ndarray): Acceleration of all particles due to Triaxial NFW external potential.
            - Potential (jnp.ndarray): Potential energy of all particles. Returned only if return_potential is True.
    """
    params_TNFW = params.TriaxialNFW_params
    M = params_TNFW.Mvir
    r_s = params_TNFW.r_s
    q1 = params_TNFW.q1  # y-axis flattening
    q2 = params_TNFW.q2  # z-axis flattening
    
    # Gauss-Legendre quadrature (order 50)
    integration_order = 50
    x_, w_ = np.polynomial.legendre.leggauss(integration_order)
    x_gl, w_gl = jnp.asarray(x_, dtype=float), jnp.asarray(w_, dtype=float)
    # Change interval from [-1, 1] to [0, 1]
    x_gl = 0.5 * (x_gl + 1)
    w_gl = 0.5 * w_gl
    
    # Central density: rho_0 = M / (4 * pi * r_s^3)
    rho0 = M / (4 * jnp.pi * r_s**3)
    
    q1sq = q1**2
    q2sq = q2**2
    
    @jit
    def ellipsoid_surface(pos, s2):
        """Compute xi^2 on the ellipsoid surface."""
        # xi^2(tau) = x^2/(1+tau) + y^2/(q1^2+tau) + z^2/(q2^2+tau)
        # with tau = 1/s^2 - 1, this becomes:
        return s2 * (
            pos[0]**2 
            + pos[1]**2 / (1 + (q1sq - 1) * s2)
            + pos[2]**2 / (1 + (q2sq - 1) * s2)
        )
    
    @jit
    def potential_single(pos):
        """Compute potential for a single particle."""
        def integrand(s):
            s2 = s**2
            xi = jnp.sqrt(ellipsoid_surface(pos, s2)) / r_s
            delta_psi_factor = 2.0 / (1.0 + xi)
            denom = jnp.sqrt(((q1sq - 1) * s2 + 1) * ((q2sq - 1) * s2 + 1))
            return delta_psi_factor / denom
        
        # Gauss-Legendre integration
        integral = jnp.sum(w_gl * vmap(integrand)(x_gl))
        
        return -2.0 * jnp.pi * params.G * rho0 * r_s**2 * q1 * q2 * integral
    
    @jit
    def acceleration_single(pos):
        """Compute acceleration for a single particle via gradient of potential."""
        return -jax.grad(potential_single)(pos)
    
    pos = state[:, 0]
    acc = vmap(acceleration_single)(pos)
    
    if return_potential:
        pot = vmap(potential_single)(pos)
        return acc, pot
    else:
        return acc
    



@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
def Thin_DoubleExponentialDiskPotential(state: jnp.ndarray,
                                        config: SimulationConfig,
                                        params: SimulationParams,
                                        return_potential=False):
    """
    Compute acceleration and potential of all particles due to a double exponential disk.

    Inspired by: https://github.com/jobovy/galpy/blob/main/galpy/potential/DoubleExponentialDiskPotential.py#L24-L429

    Args:
        state (jnp.ndarray): (N_particles, 2, 3) positions and velocities.
        config (NamedTuple): Configuration parameters (should include de_n, de_h).
        params (NamedTuple): Simulation parameters (should include amp, hr, hz).
        return_potential (bool): If True, also returns the potential.

    Returns:
        jnp.ndarray: Acceleration (N_particles, 3)
        jnp.ndarray: Potential (N_particles,) if return_potential is True
    """
    # Physical parameters
    params_ThinDoubleExpDisk = params.Thin_DoubleExpDisk_params
    Sigma = params_ThinDoubleExpDisk.Sigma
    hr = params_ThinDoubleExpDisk.hr
    hz = params_ThinDoubleExpDisk.hz
    amp = Sigma/(2*hz)

    # Integration parameters
    de_n = config.de_n
    de_h = config.de_h

    # Precompute zeros and weights (Ogata's Bessel integral formula)
    j0_zeros = jnp.array([scipy.special.jn_zeros(0, de_n) / np.pi])
    j1_zeros = jnp.array([scipy.special.jn_zeros(1, de_n) / np.pi])

    def _de_psi(x):
        return x * jnp.tan(jnp.pi * x / 2)

    def _de_psiprime(x):
        return jnp.tan(jnp.pi * x / 2) + (jnp.pi * x / 2) * (1 / jnp.cos(jnp.pi * x / 2))**2

    j0_xs = jnp.pi / de_h * _de_psi(de_h * j0_zeros)
    j0_weights = (
        2.0
        / (jnp.pi * j0_zeros * jnp.array([j1(jnp.pi * j0_zeros)]) ** 2.0)
        * jnp.array([(j0_xs)])
        * _de_psiprime(de_h * j0_zeros)
    )
    j1_xs = jnp.pi / de_h * _de_psi(de_h * j1_zeros)
    j1_weights = (
        2.0
        / (jnp.pi * j1_zeros * jnp.array([jv(2, jnp.pi * j1_zeros)]) ** 2.0)
        * jnp.array([j1(j1_xs)])
        * _de_psiprime(de_h * j1_zeros)
    )

    alpha = 1.0 / hr
    beta = 1.0 / hz

    pos = state[:, 0]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = jnp.sqrt(x**2 + y**2)
    absz = jnp.abs(z)

    # Potential
    def potential_single(R, z):
        # Avoid division by zero
        R = jnp.where(R == 0, 1e-12, R)
        def fun(x):
            denom = beta**2 - (x / R) ** 2
            numer = (
                beta * jnp.exp(-x / R * jnp.abs(z))
                - x / R * jnp.exp(-beta * jnp.abs(z))
            )
            return (alpha**2 + (x / R) ** 2) ** -1.5 * numer / denom
        # Vectorize over xs
        integrand = fun(j0_xs)
        s = -4.0 * jnp.pi * alpha / R * jnp.nansum(integrand * j0_weights)
        return s

    # Radial force
    def Rforce_single(R, z):
        R = jnp.where(R == 0, 1e-12, R)
        def fun(x):
            denom = beta**2 - (x / R) ** 2
            numer = (
                beta * jnp.exp(-x / R * jnp.abs(z))
                - x / R * jnp.exp(-beta * jnp.abs(z))
            )
            return x * (alpha**2 + (x / R) ** 2) ** -1.5 * numer / denom
        integrand = fun(j1_xs)
        s = -4.0 * jnp.pi * alpha / R**2 * jnp.nansum(integrand * j1_weights)
        return s

    # Vertical force
    def zforce_single(R, z):
        R = jnp.where(R == 0, 1e-12, R)
        def fun(x):
            denom = beta**2 - (x / R) ** 2
            numer = (
                jnp.exp(-x / R * jnp.abs(z))
                - jnp.exp(-beta * jnp.abs(z))
            )
            return (alpha**2 + (x / R) ** 2) ** -1.5 * x / R * numer / denom
        integrand = fun(j0_xs)
        s = -4.0 * jnp.pi * alpha * beta / R * jnp.nansum(integrand * j0_weights)
        return jnp.where(z > 0, s, -s)

    # Vectorize for all particles
    potential_v = jax.vmap(potential_single)
    Rforce_v = jax.vmap(Rforce_single)
    zforce_v = jax.vmap(zforce_single)

    pot = amp * potential_v(R, z)
    fR = amp * Rforce_v(R, z)
    fz = amp * zforce_v(R, z)

    # Convert to Cartesian acceleration
    with jax.ensure_compile_time_eval():
        cosphi = jnp.where(R > 0, x / R, 1.0)
        sinphi = jnp.where(R > 0, y / R, 0.0)
    ax = fR * cosphi
    ay = fR * sinphi
    az = fz

    acc = jnp.stack([ax, ay, az], axis=1)

    if return_potential:
        return acc, pot
    else:
        return acc
    

@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
def Thick_DoubleExponentialDiskPotential(state: jnp.ndarray,
                                        config: SimulationConfig,
                                        params: SimulationParams,
                                        return_potential=False):
    """
    Compute acceleration and potential of all particles due to a double exponential disk.

    Inspired by: https://github.com/jobovy/galpy/blob/main/galpy/potential/DoubleExponentialDiskPotential.py#L24-L429

    Args:
        state (jnp.ndarray): (N_particles, 2, 3) positions and velocities.
        config (NamedTuple): Configuration parameters (should include de_n, de_h).
        params (NamedTuple): Simulation parameters (should include amp, hr, hz).
        return_potential (bool): If True, also returns the potential.

    Returns:
        jnp.ndarray: Acceleration (N_particles, 3)
        jnp.ndarray: Potential (N_particles,) if return_potential is True
    """
    # Physical parameters
    params_ThickDoubleExpDisk = params.Thick_DoubleExpDisk_params
    Sigma = params_ThickDoubleExpDisk.Sigma
    hr = params_ThickDoubleExpDisk.hr
    hz = params_ThickDoubleExpDisk.hz
    amp = Sigma/(2*hz)

    # Integration parameters
    de_n = config.de_n
    de_h = config.de_h

    # Precompute zeros and weights (Ogata's Bessel integral formula)
    j0_zeros = jn_zeros(0, de_n) / jnp.pi
    j1_zeros = jn_zeros(1, de_n) / jnp.pi

    def _de_psi(x):
        return x * jnp.tan(jnp.pi * x / 2)

    def _de_psiprime(x):
        return jnp.tan(jnp.pi * x / 2) + (jnp.pi * x / 2) * (1 / jnp.cos(jnp.pi * x / 2))**2

    j0_xs = jnp.pi / de_h * _de_psi(de_h * j0_zeros)
    j0_weights = (
        2.0
        / (jnp.pi * j0_zeros * jnp.array([j1(jnp.pi * j0_zeros)]) ** 2.0)
        * jnp.array([(j0_xs)])
        * _de_psiprime(de_h * j0_zeros)
    )
    j1_xs = jnp.pi / de_h * _de_psi(de_h * j1_zeros)
    j1_weights = (
        2.0
        / (jnp.pi * j1_zeros * jnp.array([jv(2, jnp.pi * j1_zeros)])  ** 2.0)
        * jnp.array([j1(j1_xs)])
        * _de_psiprime(de_h * j1_zeros)
    )

    alpha = 1.0 / hr
    beta = 1.0 / hz

    pos = state[:, 0]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = jnp.sqrt(x**2 + y**2)
    absz = jnp.abs(z)

    # Potential
    def potential_single(R, z):
        # Avoid division by zero
        R = jnp.where(R == 0, 1e-12, R)
        def fun(x):
            denom = beta**2 - (x / R) ** 2
            numer = (
                beta * jnp.exp(-x / R * jnp.abs(z))
                - x / R * jnp.exp(-beta * jnp.abs(z))
            )
            return (alpha**2 + (x / R) ** 2) ** -1.5 * numer / denom
        # Vectorize over xs
        integrand = fun(j0_xs)
        s = -4.0 * jnp.pi * alpha / R * jnp.nansum(integrand * j0_weights)
        return s

    # Radial force
    def Rforce_single(R, z):
        R = jnp.where(R == 0, 1e-12, R)
        def fun(x):
            denom = beta**2 - (x / R) ** 2
            numer = (
                beta * jnp.exp(-x / R * jnp.abs(z))
                - x / R * jnp.exp(-beta * jnp.abs(z))
            )
            return x * (alpha**2 + (x / R) ** 2) ** -1.5 * numer / denom
        integrand = fun(j1_xs)
        s = -4.0 * jnp.pi * alpha / R**2 * jnp.nansum(integrand * j1_weights)
        return s

    # Vertical force
    def zforce_single(R, z):
        R = jnp.where(R == 0, 1e-12, R)
        def fun(x):
            denom = beta**2 - (x / R) ** 2
            numer = (
                jnp.exp(-x / R * jnp.abs(z))
                - jnp.exp(-beta * jnp.abs(z))
            )
            return (alpha**2 + (x / R) ** 2) ** -1.5 * x / R * numer / denom
        integrand = fun(j0_xs)
        s = -4.0 * jnp.pi * alpha * beta / R * jnp.nansum(integrand * j0_weights)
        return jnp.where(z > 0, s, -s)

    # Vectorize for all particles
    potential_v = jax.vmap(potential_single)
    Rforce_v = jax.vmap(Rforce_single)
    zforce_v = jax.vmap(zforce_single)

    pot = amp * potential_v(R, z)
    fR = amp * Rforce_v(R, z)
    fz = amp * zforce_v(R, z)

    # Convert to Cartesian acceleration
    with jax.ensure_compile_time_eval():
        cosphi = jnp.where(R > 0, x / R, 1.0)
        sinphi = jnp.where(R > 0, y / R, 0.0)
    ax = fR * cosphi
    ay = fR * sinphi
    az = fz

    acc = jnp.stack([ax, ay, az], axis=1)

    if return_potential:
        return acc, pot
    else:
        return acc