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

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from unxt import Quantity

# Pack all differentiable parameters into a single flat dict of plain JAX scalars
galax_params = {
    'm_Triaxial_halo': jnp.array(1e12),   # Msun
    'r_Triaxial_halo': jnp.array(20.0),   # kpc
    'q2_Triaxial_halo': jnp.array(0.9),
    'rho_thin_disk':    jnp.array(0.1),   # Msun/pc^3
    'hr_thin_disk':     jnp.array(3.0),   # kpc
    'hz_thin_disk':     jnp.array(0.3),   # kpc
    'rho_thick_disk':   jnp.array(0.01),  # Msun/pc^3
    'hr_thick_disk':    jnp.array(3.0),   # kpc
    'hz_thick_disk':    jnp.array(0.9),   # kpc
    'm_bulge':          jnp.array(1e10),  # Msun
    'r_bulge':          jnp.array(0.5),   # kpc
    'alpha_bulge':      jnp.array(1.8),
    # q1 is fixed at 1.0 — kept out of the differentiable dict
}

def make_galax_pot_from_dict(p):
    """Build a galax CompositePotential from a flat dict of JAX scalars."""
    thin_disk_mass  = 4 * jnp.pi * p['rho_thin_disk']  * (u.Msun / u.pc**3).to(u.Msun/u.kpc**3) * p['hr_thin_disk']**2  * p['hz_thin_disk']
    thick_disk_mass = 4 * jnp.pi * p['rho_thick_disk'] * (u.Msun / u.pc**3).to(u.Msun/u.kpc**3) * p['hr_thick_disk']**2 * p['hz_thick_disk']

    return gp.CompositePotential(
        halo=gp.TriaxialNFWPotential(
            m=Quantity(p['m_Triaxial_halo'], "Msun"),
            r_s=Quantity(p['r_Triaxial_halo'], "kpc"),
            q1=1.0,                           # fixed
            q2=p['q2_Triaxial_halo'],
            units="galactic",
        ),
        thin_disk=gp.MN3ExponentialPotential(
            m_tot=Quantity(thin_disk_mass, "Msun"),
            h_R=Quantity(p['hr_thin_disk'], "kpc"),
            h_z=Quantity(p['hz_thin_disk'], "kpc"),
            units="galactic",
            positive_density=True,
        ),
        thick_disk=gp.MN3ExponentialPotential(
            m_tot=Quantity(thick_disk_mass, "Msun"),
            h_R=Quantity(p['hr_thick_disk'], "kpc"),
            h_z=Quantity(p['hz_thick_disk'], "kpc"),
            units="galactic",
            positive_density=True,
        ),
        bulge=gp.PowerLawCutoffPotential(
            m_tot=Quantity(p['m_bulge'], "Msun"),
            r_c=Quantity(p['r_bulge'], "kpc"),
            alpha=p['alpha_bulge'],
            units="galactic",
        ),
    )

def galax_vcirc_from_dict(p, xyz_kpc):
    pot = make_galax_pot_from_dict(p)
    w = gc.PhaseSpaceCoordinate(
        q=Quantity([xyz_kpc[0], xyz_kpc[1], xyz_kpc[2]], "kpc"),
        p=Quantity([0.0, 0.0, 0.0], "km/s"),
        t=Quantity(0.0, "Gyr"),
    )
    return jnp.squeeze(pot.local_circular_velocity(w).value)  * (u.kpc/u.Myr).to(u.km/u.s)# 0-d → scalar

# ---- Usage ----

xyz_kpc = jnp.array([8.0, 0.0, 0.0])

# Forward pass
vcirc = galax_vcirc_from_dict(galax_params, xyz_kpc)
print(f"v_circ = {vcirc:.4f} ")

# Gradient w.r.t. ALL parameters at once (same dict structure as galax_params)
grad_dict = jax.grad(galax_vcirc_from_dict)(galax_params, xyz_kpc)

print("\nd(v_circ)/d(param):")
for k, v in grad_dict.items():
    print(f"  {k:25s} = {v:.6e}")

# Or gradient w.r.t. a scalar loss over multiple positions
xyz_batch = jnp.array([
    [8.0, 0.0, 0.0],
    # [0.0, 8.0, 0.0],
    # [6.0, 4.0, 1.0],
])
target_kms = 220.0

def loss(p):
    vc = jax.vmap(lambda xyz: galax_vcirc_from_dict(p, xyz))(xyz_batch)
    return jnp.sum((vc - target_kms)**2)

grad_loss = jax.grad(loss)(galax_params)
print("\nd(loss)/d(param):")
for k, v in grad_loss.items():
    print(f"  {k:25s} = {v:.6e}")