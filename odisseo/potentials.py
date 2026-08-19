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
                          lambda state: Thin_MN3DiskPotential(state, config=config, params=params, return_potential=True),
                          lambda state: Thick_MN3DiskPotential(state, config=config, params=params, return_potential=True),
                          lambda state: TwoPowerTriaxialPotential(state, config=config, params=params, return_potential=True),

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
                          lambda state: Thin_MN3DiskPotential(state, config=config, params=params, return_potential=False),
                          lambda state: Thick_MN3DiskPotential(state, config=config, params=params, return_potential=False),
                          lambda state: TwoPowerTriaxialPotential(state, config=config, params=params, return_potential=False),
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


    pos = state[:, 0]
    acc = acceleration(pos)

    if return_potential:
        pot = potential(pos)
        return acc, pot
    else:
        return acc
    
@partial(jax.jit, static_argnames=['return_potential'])
@jaxtyped(typechecker=typechecker)
def call_MyamotoNagai(state: jnp.ndarray, 
                        M: Union[float, jnp.ndarray],
                        a: Union[float, jnp.ndarray],
                        b: Union[float, jnp.ndarray],
                        params: SimulationParams,
                        return_potential=False):
    """
    Compute acceleration of all particles due to a MyamotoNagai disk profile. It is used as base function for MN3 approximation of douoble exponential disk.
    This function exposes directly the a, b and M parameters intstead of calling the params of the simulation

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
    
    z2 = state[:, 0, 2]**2

    Dz = (a+(z2+b**2)**0.5)
    D = jnp.linalg.norm(state[:, 0, :2], axis=1)**2 + Dz**2
    K = - params.G * M / D**(3/2)

    @jit
    def acceleration(pos):
        ax = K * pos[:, 0]
        ay = K * pos[:, 1]
        az = K * pos[:, 2] * Dz / (z2 + b**2)**0.5
        return jnp.stack([ax, ay, az], axis=1)

    @jit
    def potential(pos):
        return - params.G * M / jnp.sqrt(D)


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
    integration_order = config.glorder
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
def Thin_MN3DiskPotential(state: jnp.ndarray,
                            config: SimulationConfig,
                            params: SimulationParams,
                            return_potential=False):
    """
    Compute acceleration and potential of all particles due to a thin disk approximated by 3 Miyamoto-Nagai potentials.
    Inspired by: https://gala.adrian.pw/en/latest/_modules/gala/potential/potential/builtin/core.html#MN3ExponentialDiskPotential.
    Original paper: `Smith et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract>`

    Args:
        state (jnp.ndarray): (N_particles, 2, 3) positions and velocities.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool): If True, also returns the potential.

    Returns:
        jnp.ndarray: Acceleration (N_particles, 3)
        jnp.ndarray: Potential (N_particles,) if return_potential is True

    """
    params_ThinMN3Disk = params.ThinMN3Disk_params
    m = params_ThinMN3Disk.M
    h_R = params_ThinMN3Disk.hr
    h_z = params_ThinMN3Disk.hz
    hzR = h_z / h_R
    sech2_z = config.sech2_z
    MN3_positive_density = config.MN3_positive_density

    _K_pos_dens = jnp.array(
        [
            [0.0036, -0.0330, 0.1117, -0.1335, 0.1749],
            [-0.0131, 0.1090, -0.3035, 0.2921, -5.7976],
            [-0.0048, 0.0454, -0.1425, 0.1012, 6.7120],
            [-0.0158, 0.0993, -0.2070, -0.7089, 0.6445],
            [-0.0319, 0.1514, -0.1279, -0.9325, 2.6836],
            [-0.0326, 0.1816, -0.2943, -0.6329, 2.3193],
        ]
    )
    _K_neg_dens = jnp.array(
        [
            [-0.0090, 0.0640, -0.1653, 0.1164, 1.9487],
            [0.0173, -0.0903, 0.0877, 0.2029, -1.3077],
            [-0.0051, 0.0287, -0.0361, -0.0544, 0.2242],
            [-0.0358, 0.2610, -0.6987, -0.1193, 2.0074],
            [-0.0830, 0.4992, -0.7967, -1.2966, 4.4441],
            [-0.0247, 0.1718, -0.4124, -0.5944, 0.7333],
        ]
    )

    K = jnp.where(MN3_positive_density, _K_pos_dens, _K_neg_dens)
    b_hR = jnp.where(sech2_z, -0.033 * hzR**3 + 0.262 * hzR**2 + 0.659 * hzR, -0.269 * hzR**3 + 1.08 * hzR**2 + 1.092 * hzR)
    x = jnp.vander(jnp.array([b_hR]), N=5)[0]

    param_vec = K @ x

    _ms = param_vec[:3] * m
    _as = param_vec[3:] * h_R
    _b = b_hR * h_R
    _b = jnp.broadcast_to(_b, _ms.shape) #needed for vmap


    c_only = {}
    for i in range(3):
        c_only[f"m{i + 1}"] = _ms[i]
        c_only[f"a{i + 1}"] = _as[i]
        c_only[f"b{i + 1}"] = _b
    
    acc_total = jax.vmap(lambda m, a, b: call_MyamotoNagai(state, m, a, b, params, return_potential=False))(
        _ms, _as, _b
    ).sum(axis=0)

    if return_potential:
        pot_total = jax.vmap(lambda m, a, b: call_MyamotoNagai(state, m, a, b, params, return_potential=True))(
            _ms, _as, _b
        )[1].sum(axis=0)
        return acc_total, pot_total
    else:
        return acc_total



@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
def Thick_MN3DiskPotential(state: jnp.ndarray,
                            config: SimulationConfig,
                            params: SimulationParams,
                            return_potential=False):
    """
    Compute acceleration and potential of all particles due to a thin disk approximated by 3 Miyamoto-Nagai potentials.
    Inspired by: https://gala.adrian.pw/en/latest/_modules/gala/potential/potential/builtin/core.html#MN3ExponentialDiskPotential.
    Original paper: `Smith et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract>`

    Args:
        state (jnp.ndarray): (N_particles, 2, 3) positions and velocities.
        config (NamedTuple): Configuration parameters.
        params (NamedTuple): Simulation parameters.
        return_potential (bool): If True, also returns the potential.

    Returns:
        jnp.ndarray: Acceleration (N_particles, 3)
        jnp.ndarray: Potential (N_particles,) if return_potential is True

    """
    params_ThickMN3Disk = params.ThickMN3Disk_params
    m = params_ThickMN3Disk.M
    h_R = params_ThickMN3Disk.hr
    h_z = params_ThickMN3Disk.hz
    hzR = h_z / h_R
    sech2_z = config.sech2_z
    MN3_positive_density = config.MN3_positive_density

    _K_pos_dens = jnp.array(
        [
            [0.0036, -0.0330, 0.1117, -0.1335, 0.1749],
            [-0.0131, 0.1090, -0.3035, 0.2921, -5.7976],
            [-0.0048, 0.0454, -0.1425, 0.1012, 6.7120],
            [-0.0158, 0.0993, -0.2070, -0.7089, 0.6445],
            [-0.0319, 0.1514, -0.1279, -0.9325, 2.6836],
            [-0.0326, 0.1816, -0.2943, -0.6329, 2.3193],
        ]
    )
    _K_neg_dens = jnp.array(
        [
            [-0.0090, 0.0640, -0.1653, 0.1164, 1.9487],
            [0.0173, -0.0903, 0.0877, 0.2029, -1.3077],
            [-0.0051, 0.0287, -0.0361, -0.0544, 0.2242],
            [-0.0358, 0.2610, -0.6987, -0.1193, 2.0074],
            [-0.0830, 0.4992, -0.7967, -1.2966, 4.4441],
            [-0.0247, 0.1718, -0.4124, -0.5944, 0.7333],
        ]
    )

    K = jnp.where(MN3_positive_density, _K_pos_dens, _K_neg_dens)
    b_hR = jnp.where(sech2_z, -0.033 * hzR**3 + 0.262 * hzR**2 + 0.659 * hzR, -0.269 * hzR**3 + 1.08 * hzR**2 + 1.092 * hzR)
    x = jnp.vander(jnp.array([b_hR]), N=5)[0]

    param_vec = K @ x

    _ms = param_vec[:3] * m
    _as = param_vec[3:] * h_R
    _b = b_hR * h_R
    _b = jnp.broadcast_to(_b, _ms.shape)
    
    acc_total = jax.vmap(lambda m, a, b: call_MyamotoNagai(state, m, a, b, params, return_potential=False))(
        _ms, _as, _b
    ).sum(axis=0)

    if return_potential:
        pot_total = jax.vmap(lambda m, a, b: call_MyamotoNagai(state, m, a, b, params, return_potential=True))(
            _ms, _as, _b
        )[1].sum(axis=0)
        return acc_total, pot_total
    else:
        return acc_total


@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
def TwoPowerTriaxialPotential(state: jnp.ndarray,
                              config: SimulationConfig,
                              params: SimulationParams,
                              return_potential=False):
    """
    General triaxial two-power-law potential:
        rho(x,y,z) = amp/(4*pi*a^3) * 1/(m/a)^alpha * 1/(1+m/a)^(beta-alpha)
        m^2 = x^2 + y^2/b^2 + z^2/c^2

    Args:
        state: (N_particles, 2, 3) positions and velocities.
        config: Configuration parameters.
        params: Simulation parameters.
        return_potential: If True, also returns the potential.

    Returns:
        acc: (N_particles, 3) acceleration
        pot: (N_particles,) potential (if return_potential)
    """
    p = params.TwoPowerTriaxial_params
    rho = p.rho
    a = p.a
    alpha = p.alpha
    beta = p.beta
    b = p.b
    c = p.c

    b2 = b**2
    c2 = c**2

    # Gauss-Legendre quadrature (order 50)
    glorder = config.glorder
    x_, w_ = np.polynomial.legendre.leggauss(glorder)
    x_gl, w_gl = jnp.asarray(x_, dtype=float), jnp.asarray(w_, dtype=float)
    x_gl = 0.5 * (x_gl + 1)
    w_gl = 0.5 * w_gl

    # Normalization
    # norm = params.G * rho / (4 * jnp.pi * a**3) # we use directly the normalization rho
    # norm = params.G *rho 
    norm = params.G * rho 

    def safe_hyp2f1(a, b, c, z):
        # Transformation: z_new = z / (z - 1)
        # This maps z in (-inf, -1] to z_new in [0.5, 1)
        z_new = z / (z - 1.0)
        transformed_val = jnp.pow(1.0 - z, -a) * jsp.hyp2f1(a, c - b, c, z_new)
        
        # Use jnp.where to choose the transformation only when z < -0.9
        # (We use -0.9 to stay well away from the boundary of the unit circle)
        return jnp.where(z < -0.9, transformed_val, jsp.hyp2f1(a, b, c, z))

    @jit
    def mfunc(pos):
        return jnp.sqrt(pos[0]**2 + pos[1]**2 / b2 + pos[2]**2 / c2)

    @jit
    def _psi_inf():
        # psi_inf = gamma(beta-2) * gamma(3-alpha) / gamma(beta-alpha)
        return jsp.gamma(beta - 2.0) * jsp.gamma(3.0 - alpha) / jsp.gamma(beta - alpha)

    psi_inf = _psi_inf()
    twominusalpha = 2.0 - alpha
    threeminusalpha = 3.0 - alpha
    betaminusalpha = beta - alpha

    @jit
    def psi(m):
        # See galpy: _psi
        # If twominusalpha == 0:
        #   -2 a^2 (a/m)^(beta-alpha) / (beta-alpha) * hyp2f1(b-a, b-a, b-a+1, -a/m)
        # else:
        #   -2 a^2 [psi_inf - (m/a)^(2-alpha)/(2-alpha) * hyp2f1(2-alpha, beta-alpha, 3-alpha, -m/a)]
        # val_z = -m / a
        # # This will print every time the JIT-compiled function is executed
        # jax.debug.print("Current m: {m_val}, Argument z: {z_val}", m_val=m, z_val=val_z)

        # res = jsp.hyp2f1(twominusalpha, betaminusalpha, threeminusalpha, val_z)
    
        # # Check for NaNs or Infs
        # jax.debug.print("hyp2f1 result: {r}", r=res)

        def branch():
            # return -2.0 * a**2 * (a / m) ** betaminusalpha / betaminusalpha * jsp.hyp2f1(
            #     betaminusalpha, betaminusalpha, betaminusalpha + 1, -a / m
            # )
            return (
                -2.0
                * a**2
                * (a / m) ** betaminusalpha
                / betaminusalpha
                # * jsp.hyp2f1(
                * safe_hyp2f1(
                    betaminusalpha,
                    betaminusalpha,
                    betaminusalpha + 1,
                    -a / m,
                )
            )
        def main():
            # return -2.0 * a**2 * (
            #     psi_inf
            #     - (m / a) ** twominusalpha / twominusalpha * jsp.hyp2f1(
            #         twominusalpha, betaminusalpha, threeminusalpha, -m / a
            #     )
            # )
            return (
                -2.0
                * a**2
                * (
                    psi_inf
                    - (m / a) ** twominusalpha
                    / twominusalpha
                    # * jsp.hyp2f1(
                    * safe_hyp2f1(
                        twominusalpha,
                        betaminusalpha,
                        threeminusalpha,
                        -m / a,
                    )
                )
            )
        return jax.lax.cond(jnp.abs(twominusalpha) < 1e-10, branch, main)

    @jit
    def dens(m):
        return (a / m) ** alpha / (1.0 + m / a) ** betaminusalpha

    @jit
    def force_integral(pos, i):
        # Integrate over s in [0,1]
        def integrand(s):
            t = 1.0 / s**2 - 1.0
            m = jnp.sqrt(
                pos[0] ** 2 / (1.0 + t)
                + pos[1] ** 2 / (b2 + t)
                + pos[2] ** 2 / (c2 + t)
            )
            numer = (
                pos[0] / (1.0 + t) * (i == 0)
                + pos[1] / (b2 + t) * (i == 1)
                + pos[2] / (c2 + t) * (i == 2)
            )
            denom = jnp.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            return dens(m) * numer / denom
        return jnp.sum(w_gl * jax.vmap(integrand)(x_gl))

    @jit
    def potential_integral(pos):
        def integrand(s):
            t = 1.0 / s**2 - 1.0
            # m = jnp.sqrt(
            #     pos[0] ** 2 / (1.0 + t)
            #     + pos[1] ** 2 / (b2 + t)
            #     + pos[2] ** 2 / (c2 + t)
            # )
            # denom = jnp.sqrt((1.0 + (b2 - 1.0) * s**2) * (1.0 + (c2 - 1.0) * s**2))
            # return psi(m) / denom
            return psi(
                    jnp.sqrt(pos[0]**2.0 / (1.0 + t) + pos[1]**2.0 / (b2 + t) + pos[2]**2.0 / (c2 + t))
                ) / jnp.sqrt((1.0 + (b2 - 1.0) * s**2.0) * (1.0 + (c2 - 1.0) * s**2.0))
        return jnp.sum(w_gl * jax.vmap(integrand)(x_gl))

    @jit
    def acc_and_pot_single(pos):
        acc = -4.0 * jnp.pi * b * c * norm * jnp.array(                                  #is norm correect here?
        # acc = -4.0 *  jnp.pi * b * c *  jnp.array(                           
            [force_integral(pos, 0), force_integral(pos, 1), force_integral(pos, 2)] 
        )
        pot = 2.0 * jnp.pi * b * c * norm * potential_integral(pos)                     #is norm correect here? 
        # pot = 2.0 * jnp.pi * b * c * potential_integral(pos)             
        # pot = potential_integral(pos)  
        return acc, pot

    pos = state[:, 0]
    acc, pot = jax.vmap(acc_and_pot_single)(pos)

    if return_potential:
        return acc, pot
    else:
        return acc


