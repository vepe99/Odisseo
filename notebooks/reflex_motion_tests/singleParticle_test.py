print("Running reflex motion test")

# Only make a single GPU visible
from autocvd import autocvd
autocvd(num_gpus = 1)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import jax.numpy as jnp
from jax import random

import plotly.graph_objects as go

from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, HernquistParams, NFWParams, MNParams, PSPParams, PointMassParams, DynamicalFrictionParams
from odisseo.option_classes import NFW_POTENTIAL, DIFFRAX_BACKEND, MN_POTENTIAL, PSP_POTENTIAL, HERNQUIST_POTENTIAL, DOPRI8
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.initial_condition import ic_two_body

print("Imports successful")

# Setup
code_length = 10 * u.kpc
code_mass = 1e11 * u.Msun
code_time = 1 * u.Gyr
code_units = CodeUnits(unit_length = code_length, 
                       unit_mass = code_mass,
                       unit_time = code_time,
                       G = None
)  

config = SimulationConfig(N_particles = 1, 
                          return_snapshots = True, 
                          num_snapshots = 1000, 
                          fixed_timestep = False,
                          num_timesteps = 1000,
                          softening = 0.1 * u.pc.to(code_units.code_length),
                          integrator = DIFFRAX_BACKEND,
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          external_accelerations = ((NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL),(HERNQUIST_POTENTIAL,)), 
                          # external_accelerations = ((POINT_MASS,),(HERNQUIST_POTENTIAL,)), 
                          reflex_motion = True,
                          diffrax_solver = DOPRI8
)       

params = SimulationParams(t_end = -1 * u.Gyr.to(code_units.code_time),
                          G = code_units.G,
                          Hernquist_params = HernquistParams(M = 15e10 * u.Msun.to(code_units.code_mass), 
                                                             r_s = 17.14 * u.kpc.to(code_units.code_length)),
                                                 # Mvir is actually M_char
                          NFW_params = NFWParams(Mvir = 436723306039.5261 * u.Msun.to(code_units.code_mass), 
                                                 r_s = 16 * u.kpc.to(code_units.code_length)),
                          MN_params = MNParams(M = 68176739700.15533 * u.Msun.to(code_units.code_mass), 
                                               a = 3.0 * u.kpc.to(code_units.code_length), 
                                               b = 0.28 * u.kpc.to(code_units.code_length)),
                          PSP_params = PSPParams(M = 4500232468.7387295 * u.Msun.to(code_units.code_mass), 
                                                 alpha = 1.8, 
                                                 r_c = 1.9 * u.kpc.to(code_units.code_length)),
                          PointMass_params = PointMassParams(M = 15e11 * u.Msun.to(code_units.code_mass)),
                          DynamicalFriction_params = DynamicalFrictionParams(sigma_MW = 120. * (u.km/u.s).to(code_units.code_velocity),
                                                                             lambda_df = 0.001,
                                                                             coulomb_log_numerator = 100. * u.kpc.to(code_units.code_length)),
) 

key = random.PRNGKey(1)

# Setup two particle system (MW and test particle) with stable orbit
particle_mass = 1e4 * u.Msun.to(code_units.code_mass)
rp = 10 * u.kpc.to(code_units.code_length)
e = 0.
M = params.NFW_params.Mvir
r_s = params.NFW_params.r_s
MW_mass_at_r = M * (jnp.log(1 + (rp / r_s)) - (rp / r_s) / (1 + (rp / r_s)))

pos, vel, mass = ic_two_body(MW_mass_at_r, particle_mass, rp, e, params)

# Setup test particle
# pos = jnp.array([[10., 0., 0.]]) * u.kpc.to(code_units.code_length)
# vel = jnp.array([[0., 100., 0.]]) * (u.km/u.s).to(code_units.code_velocity)
# particle_mass = jnp.array([1e4]) * u.Msun.to(code_units.code_mass)
particle_mass = jnp.array([mass[1]])
particle_state = construct_initial_state(jnp.array([pos[1]]), jnp.array([vel[1]])) # state is a (N_particles x 2 x 3)

# Add the Milky Way as particle in index 0
MW_pos = jnp.array([[0., 0., 0.]]) * u.kpc.to(code_units.code_length)
MW_vel = jnp.array([[0., 0., 0.]]) * (u.km/u.s).to(code_units.code_velocity)
MW_mass = jnp.array([15e11]) * u.Msun.to(code_units.code_mass)
# MW_mass = jnp.array([4.3683325e11]) * u.Msun.to(code_units.code_mass)
MW_state = construct_initial_state(MW_pos, MW_vel)

# Add the LMC as particle in index 1
LMC_pos = jnp.array([[-0.6, -41.3, -27.1]]) * u.kpc.to(code_units.code_length)
LMC_vel = jnp.array([[-63.9, -213.8, 206.6]]) * (u.km/u.s).to(code_units.code_velocity)
LMC_mass = jnp.array([15e10]) * u.Msun.to(code_units.code_mass)
# LMC_mass = jnp.array([4.3683325e11]) * u.Msun.to(code_units.code_mass)
LMC_state = construct_initial_state(LMC_pos, LMC_vel)

state = jnp.concatenate([MW_state, LMC_state, particle_state], axis=0)
mass = jnp.concatenate([MW_mass, LMC_mass, particle_mass], axis=0) 

print("Setup successful")

# Simulation
snapshots = time_integration(state, mass, config, params) # snapshots is a (N_snapshots x N_particles x 2 x 3)
print("Backward time integration successful")

# Visualization
length_scale = code_units.code_length.to(u.kpc)
time_scale = code_units.code_time.to(u.Gyr)

fig = go.Figure()

# Particle trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 2, 0, 0] * length_scale,
    y = snapshots.states[:, 2, 0, 1] * length_scale,
    z = snapshots.states[:, 2, 0, 2] * length_scale,
    mode='lines', showlegend=False, line=dict(
        color=snapshots.times * time_scale,  # time index
        colorscale='Viridis',
        reversescale=True,
        colorbar=dict(title='Time [Gyr]', len=0.5, y=0.5),
        width=4
    )
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 2, 0, 0] * length_scale],
    y = [snapshots.states[0, 2, 0, 1] * length_scale],
    z = [snapshots.states[0, 2, 0, 2] * length_scale],
    mode='markers', name='Particle', marker=dict(color='black')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 2, 0, 0] * length_scale],
    y = [snapshots.states[-1, 2, 0, 1] * length_scale],
    z = [snapshots.states[-1, 2, 0, 2] * length_scale],
    mode='markers', showlegend=False, marker=dict(color='black')
))

# MW trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 0, 0, 0] * length_scale,
    y = snapshots.states[:, 0, 0, 1] * length_scale,
    z = snapshots.states[:, 0, 0, 2] * length_scale,
    mode='lines', showlegend=False, line=dict(
        color=snapshots.times * time_scale,
        colorscale='Viridis',
        width=4
    )
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 0, 0, 0] * length_scale],
    y = [snapshots.states[0, 0, 0, 1] * length_scale],
    z = [snapshots.states[0, 0, 0, 2] * length_scale],
    mode='markers', name='MW', marker=dict(color='purple')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 0, 0, 0] * length_scale],
    y = [snapshots.states[-1, 0, 0, 1] * length_scale],
    z = [snapshots.states[-1, 0, 0, 2] * length_scale],
    mode='markers', showlegend=False, marker=dict(color='purple')
))

# LMC trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 1, 0, 0] * length_scale,
    y = snapshots.states[:, 1, 0, 1] * length_scale,
    z = snapshots.states[:, 1, 0, 2] * length_scale,
    mode='lines', showlegend=False, line=dict(
        color=snapshots.times * time_scale,
        colorscale='Viridis',
        width=4
    )
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 1, 0, 0] * length_scale],
    y = [snapshots.states[0, 1, 0, 1] * length_scale],
    z = [snapshots.states[0, 1, 0, 2] * length_scale],
    mode='markers', name='LMC', marker=dict(color='blue')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 1, 0, 0] * length_scale],
    y = [snapshots.states[-1, 1, 0, 1] * length_scale],
    z = [snapshots.states[-1, 1, 0, 2] * length_scale],
    mode='markers', showlegend=False, marker=dict(color='blue')
))

fig.update_layout(scene=dict(
    xaxis_title = 'X [kpc]',
    yaxis_title = 'Y [kpc]',
    zaxis_title = 'Z [kpc]'
))

fig.write_html("./notebooks/reflex_motion_tests/orbits_dynFric.html")