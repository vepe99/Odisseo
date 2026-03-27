import jax
import jax.numpy as jnp
from odisseo.option_classes import SimulationConfig, SimulationParams



def LMC_dynamical_friction(state: jnp.ndarray,
                           config: SimulationConfig,
                           params: SimulationParams):
    """
    Compute the dynamical friction acceleration on the LMC due to the Milky Way's dark matter halo.
    Assumes that the state of the MW is sored at index 0, and the state of the LMC is stored at index 1. 
    Assumes that the MW halo is modeled as an NFW potential, and the LMC is modeled as a Hernquist potential.

    Parameters:
    -----------
    state : jnp.ndarray (N_particles, 2, 3)
        The current state vector of all particles, containing their position and velocity.
    mass : jnp.ndarray (N_particles,)
        The mass of all particles.
    config : SimulationConfig
        The configuration of the simulation.
    params : SimulationParams
        The parameters of the simulation.
        
    Returns:
    --------
    jnp.ndarray (3,)
        The dynamical friction acceleration on the LMC.
    """

    params_df = params.DynamicalFriction_params
    params_MWhalo = params.NFW_params
    params_LMC = params.Hernquist_params

    # Actual NFW desnity at the position of the LMC
    r = jnp.linalg.norm(state[1, 0, :] - state[0, 0, :])
    r_s = params_MWhalo.r_s
    M_char = params_MWhalo.Mvir # Actually M_char, not Mvir
    rho_s = M_char / (4 * jnp.pi * r_s**3)
    rho_MW = rho_s / ((r/r_s) * (1 + (r/r_s))**2)

    vel_LMC = state[1, 1, :] - state[0, 1, :]
    v = jnp.maximum(jnp.linalg.norm(vel_LMC), config.softening)
    epsilon = 1.6 * params.Hernquist_params.r_s
    ln_lambda = jnp.log(params_df.coulomb_log_numerator/epsilon)
    X = v / (jnp.sqrt(2) * params_df.sigma_MW)

    dynamical_friction_acceleration = -4 * jnp.pi * params.G**2 * params_LMC.M * rho_MW * ln_lambda / (v**3) * (jax.scipy.special.erf(X) - 2 * X / jnp.sqrt(jnp.pi) * jnp.exp(-X**2)) * vel_LMC * params_df.lambda_df

    return dynamical_friction_acceleration