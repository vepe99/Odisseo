# from autocvd import autocvd
# autocvd(num_gpus=1)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Ensure only GPU 0 is used

from functools import partial
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from jax import jit
import jax.random as jr
import jax.numpy as jnp
import jax

from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.potentials import combined_external_acceleration_vmpa_switch
from odisseo.option_classes import SimulationConfig, SimulationParams, PlummerParams, PSPParams, TriaxialNFWParams,ThickMN3DiskParams, ThinMN3DiskParams 
from odisseo.option_classes import PSP_POTENTIAL, TRIAXIAL_NFW_POTENTIAL, THICK_MN3_DISK, THIN_MN3_DISK
from odisseo.units import CodeUnits


# ...existing code...
parameters_dict = {
    'm_Triaxial_halo': jnp.array([1e12], dtype=jnp.float32),
    'r_Triaxial_halo': jnp.array([20.0], dtype=jnp.float32),
    'q1_Triaxial_halo': jnp.array([1.0], dtype=jnp.float32),
    'q2_Triaxial_halo': jnp.array([0.9], dtype=jnp.float32),
    'rho_thin_disk': jnp.array([0.1], dtype=jnp.float32),
    'hr_thin_disk': jnp.array([3.0], dtype=jnp.float32),
    'hz_thin_disk': jnp.array([0.3], dtype=jnp.float32),
    'rho_thick_disk': jnp.array([0.01], dtype=jnp.float32),
    'hr_thick_disk': jnp.array([3.0], dtype=jnp.float32),
    'hz_thick_disk': jnp.array([0.9], dtype=jnp.float32),
    'm_bulge': jnp.array([1e10], dtype=jnp.float32),
    'r_bulge': jnp.array([0.5], dtype=jnp.float32),
    'alpha_bulge': jnp.array([1.8], dtype=jnp.float32),
}
# ...existing code...


def disk_masses_from_params(p):
    """
    Compute total masses of thin and thick disks from parameter dict.

    Args:
        p: dict of JAX arrays (same structure as parameters_dict)

    Returns:
        dict with:
            - M_thin_disk
            - M_thick_disk
    """

    def s(x):
        return jnp.squeeze(x)  # ensure scalar for clean autodiff

    # Extract parameters
    rho_thin = s(p['rho_thin_disk']) * (u.Msun / u.pc**3).to(u.Msun / u.kpc**3)
    hr_thin  = s(p['hr_thin_disk'])
    hz_thin  = s(p['hz_thin_disk'])

    rho_thick = s(p['rho_thick_disk']) * (u.Msun / u.pc**3).to(u.Msun / u.kpc**3)
    hr_thick  = s(p['hr_thick_disk'])
    hz_thick  = s(p['hz_thick_disk'])

    # Mass formula
    M_thin  = 4 * jnp.pi * rho_thin  * hr_thin**2  * hz_thin
    M_thick = 4 * jnp.pi * rho_thick * hr_thick**2 * hz_thick

    return {
        "M_thin_disk": M_thin,
        "M_thick_disk": M_thick,
    }

code_length = 1 * u.kpc
code_mass = 1 * u.Msun
G = 1
code_time = 1 * u.Myr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  


config = SimulationConfig(N_particles = 1000, 
                          return_snapshots = True, 
                          num_snapshots = 1000, 
                          num_timesteps = 1000, 
                          external_accelerations=(TRIAXIAL_NFW_POTENTIAL, THICK_MN3_DISK, THIN_MN3_DISK, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values


# Numeric conversion factors (Python floats, JAX-safe in traced code)
MSUN_TO_CODE_MASS = float(u.Msun.to(code_units.code_mass))
KPC_TO_CODE_LENGTH = float(u.kpc.to(code_units.code_length))
CODE_VEL_TO_KMS = float(code_units.code_velocity.to(u.km / u.s))


def s(x):
    """Squeeze array params to scalars for JAX grad compatibility."""
    return jnp.squeeze(x)

# ...existing code...
def construct_params_from_dict(p):
    """Build SimulationParams from a flat dict of JAX scalars."""
    disk_masses = disk_masses_from_params(p)

    return SimulationParams(
        t_end = (4 * u.Gyr).to(code_units.code_time).value,
        Plummer_params= PlummerParams(
            Mtot=(2.5e4 * u.Msun).to(code_units.code_mass).value,
            a=(8 * u.pc).to(code_units.code_length).value
        ),
        PSP_params= PSPParams(
            M     = s(p['m_bulge']) * MSUN_TO_CODE_MASS,
            alpha = s(p['alpha_bulge']),
            r_c   = s(p['r_bulge']) * KPC_TO_CODE_LENGTH
        ),
        TriaxialNFW_params= TriaxialNFWParams(
            Mvir = s(p['m_Triaxial_halo']) * MSUN_TO_CODE_MASS,
            r_s  = s(p['r_Triaxial_halo']) * KPC_TO_CODE_LENGTH,
            q1   = 1.0,
            q2   = s(p['q2_Triaxial_halo'])
        ),
        ThinMN3Disk_params= ThinMN3DiskParams(
            M  = s(disk_masses['M_thin_disk']) * MSUN_TO_CODE_MASS,
            hr = s(p['hr_thin_disk']) * KPC_TO_CODE_LENGTH,
            hz = s(p['hz_thin_disk']) * KPC_TO_CODE_LENGTH
        ),
        ThickMN3Disk_params= ThickMN3DiskParams(
            M  = s(disk_masses['M_thick_disk']) * MSUN_TO_CODE_MASS,
            hr = s(p['hr_thick_disk']) * KPC_TO_CODE_LENGTH,
            hz = s(p['hz_thick_disk']) * KPC_TO_CODE_LENGTH
        ),
        G=code_units.G,
    )
# ...existing code...



@partial(jax.jit, static_argnames=['config'])
def circular_velocity_at_xyz(xyz: jnp.ndarray,
                              config: SimulationConfig,
                              p: dict) -> jnp.ndarray:
    """
    Compute the local circular velocity at arbitrary (x, y, z) positions.

    Uses the general formula:
        v_circ = sqrt(r * |dPhi/dr|) = sqrt(r * |∇Φ · r̂|)

    where r = ||xyz|| and r̂ = xyz / r.

    Args:
        xyz: array of shape (N, 3) — positions in code units
        config: SimulationConfig (static)
        params: SimulationParams (differentiable)

    Returns:
        v_circ: array of shape (N,)
    """
    params = construct_params_from_dict(p)
    xyz = jnp.atleast_2d(xyz)           # (N, 3)
    n = xyz.shape[0]

    # Build state (N, 2, 3) with zero velocities
    state = jnp.stack([xyz, jnp.zeros_like(xyz)], axis=1)

    # acc = -∇Φ, shape (N, 3)
    acc = combined_external_acceleration_vmpa_switch(state, config, params, return_potential=False)

    # r and r̂
    r = jnp.linalg.norm(xyz, axis=-1)          # (N,)
    r_hat = xyz / r[:, None]                    # (N, 3)

    # dPhi/dr = ∇Φ · r̂ = -acc · r̂
    dPhi_dr = -jnp.sum(acc * r_hat, axis=-1)   # (N,)

    return jnp.sqrt(r * jnp.abs(dPhi_dr)) * CODE_VEL_TO_KMS


@partial(jax.jit, static_argnames=['config', 'func'])
def vcirc_func(xyz: jnp.ndarray,
               config: SimulationConfig,
               p: dict,
               func=lambda vc: vc) -> jnp.ndarray:
    """
    Evaluate an arbitrary scalar function of the circular velocity at positions xyz.

    Args:
        xyz: array of shape (N, 3) — positions in code units
        config: static config
        params: differentiable params
        func: callable applied to v_circ array — should return a scalar for grad

    Returns:
        func(v_circ(xyz))
    """
    vc = circular_velocity_at_xyz(xyz, config, p) 
    return func(vc)


grad_vcirc = jax.grad(vcirc_func, argnums=2)


# ---- Example usage ----

target = 220.0 # km/ss

# Arbitrary positions, not just in the midplane
# xyz_eval = jnp.array([
#     [(8.0 * u.kpc).to(code_units.code_length).value, 0.0, 0.0],          # on x-axis
#     [0.0, (8.0 * u.kpc).to(code_units.code_length).value, 0.0],          # on y-axis
#     [(6.0 * u.kpc).to(code_units.code_length).value,
#      (4.0 * u.kpc).to(code_units.code_length).value,
#      (1.0 * u.kpc).to(code_units.code_length).value],                    # off-plane
# ])

xyz_eval = jnp.array([8.0, 0.0, 0.0])


# 1. Circular velocity at each position
vc = circular_velocity_at_xyz(xyz_eval, config, parameters_dict)
print(f"v_circ = {vc[0]:.4f} ")


# 2. Residual from target
residual = vcirc_func(xyz_eval, config, parameters_dict, func=lambda vc: vc - target)
print("v_circ - target:", residual )

# 3. Scalar MSE loss + gradient w.r.t. params
loss_fn = lambda vc: jnp.sum((vc - target)**2)
loss = vcirc_func(xyz_eval, config, parameters_dict, func=loss_fn)
grads = grad_vcirc(xyz_eval, config, parameters_dict, func=loss_fn)

print("loss:", loss)
# print("d(loss)/d(params):")
# print('alpha_bulge:', grads.PSP_params.alpha)
# print('hr_thick_disk:', grads.ThickMN3Disk_params.hr)
# print('hr_thin_disk:', grads.ThinMN3Disk_params.hr)
# print('hz_thick_disk:', grads.ThickMN3Disk_params.hz)
# print('hz_thin_disk:', grads.ThinMN3Disk_params.hz)
# print('m_Triaxial_halo:', grads.TriaxialNFW_params.Mvir)
# print('m_bulge:', grads.PSP_params.M)
# print('q2_Triaxial_halo:', grads.TriaxialNFW_params.q2)
# print('r_Triaxial_halo:', grads.TriaxialNFW_params.r_s)
# print('r_bulge:', grads.PSP_params.r_c)
# print('rho_thick_disk:', grads.ThickMN3Disk_params.M)
# print('rho_thin_disk:', grads.ThinMN3Disk_params.M)

# ...existing code...
# print("\nd(loss)/d(param):")
for k, v in grads.items():   # was grad_loss
    print(f"  {k:25s} = {v[0]:.4e}")
