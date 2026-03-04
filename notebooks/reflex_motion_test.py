print("Running reflex motion test")

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
from odisseo.option_classes import SimulationConfig, SimulationParams, HernquistParams, NFWParams, MNParams, PSPParams, PointMassParams
from odisseo.option_classes import NFW_POTENTIAL, DIFFRAX_BACKEND, MN_POTENTIAL, PSP_POTENTIAL, HERNQUIST_POTENTIAL, TSIT5, POINT_MASS
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import energy_angular_momentum_plot

print("Imports successful")

# Setup
G = 1
code_length = 10 * u.kpc
code_mass = 1e11 * u.Msun
code_time = 1 * u.Gyr
code_units = CodeUnits(unit_length = code_length,
                       unit_mass = code_mass,
                       G = G, 
                       unit_time = code_time
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
                        # external_accelerations = ((NFW_POTENTIAL,),(NFW_POTENTIAL,)), 
                          reflex_motion = True,
                          diffrax_solver = TSIT5
)       

params = SimulationParams(t_end = -1 * u.Gyr.to(code_units.code_time),
                          G = code_units.G,
                          Hernquist_params=HernquistParams(M = 15e10 * u.Msun.to(code_units.code_mass), r_s = 17.14 * u.kpc.to(code_units.code_length)),
                          NFW_params=NFWParams(Mvir = 4.3683325e11 * u.Msun.to(code_units.code_mass), r_s = 15.3 * u.kpc.to(code_units.code_length)),
                          MN_params=MNParams(M = 68_193_902_782.346756 * u.Msun.to(code_units.code_mass), a = 3.0 * u.kpc.to(code_units.code_length), b = 0.28 * u.kpc.to(code_units.code_length)),
                          PSP_params=PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass), alpha = 1.8, r_c = 1.9 * u.kpc.to(code_units.code_length)),
                          PointMass_params=PointMassParams(M = 15e10 * u.Msun.to(code_units.code_mass))
) 

key = random.PRNGKey(1)

# State of test particle
# pos = jnp.array([[10., 0., 0.]]) * u.kpc.to(code_units.code_length)
# vel = jnp.array([[0., 100., 0.]]) * (u.km/u.s).to(code_units.code_velocity)
# particle_mass = jnp.array([1e4]) * u.Msun.to(code_units.code_mass)
# particle_state = construct_initial_state(pos, vel) # state is a (N_particles x 2 x 3)

# Add the Milky Way as particle in index 0
MW_pos = jnp.array([[0., 0., 0.]]) * u.kpc.to(code_units.code_length)
MW_vel = jnp.array([[0., 0., 0.]]) * (u.km/u.s).to(code_units.code_velocity)
MW_mass = jnp.array([15e11]) * u.Msun.to(code_units.code_mass)
# MW_mass = jnp.array([4.3683325e11]) * u.Msun.to(code_units.code_mass)
MW_state = construct_initial_state(MW_pos, MW_vel) # state is a (N_particles x 2 x 3)

# Add the LMC as particle in index 1
LMC_pos = jnp.array([[-0.6, -41.3, -27.1]]) * u.kpc.to(code_units.code_length)
LMC_vel = jnp.array([[-63.9, -213.8, 206.6]]) * (u.km/u.s).to(code_units.code_velocity)
LMC_mass = jnp.array([15e10]) * u.Msun.to(code_units.code_mass)
# LMC_mass = jnp.array([4.3683325e11]) * u.Msun.to(code_units.code_mass)
LMC_state = construct_initial_state(LMC_pos, LMC_vel) # state is a (N_particles x 2 x 3)

final_state = jnp.concatenate([MW_state, LMC_state], axis=0)
final_mass = jnp.concatenate([MW_mass, LMC_mass], axis=0) 

print("Setup successful")

# Simulation
snapshots = time_integration(final_state, final_mass, config, params) # snapshots is a NameTuple with states=(N_snapshots x N_particles x 2 x 3) array

print("Time integration successful")

# Visualization
scale = code_units.code_length.to(u.kpc)

fig = go.Figure()

# Particle trajectory
# fig.add_trace(go.Scatter3d(
#     x = snapshots.states[:, 2, 0, 0] * scale,
#     y = snapshots.states[:, 2, 0, 1] * scale,
#     z = snapshots.states[:, 2, 0, 2] * scale,
#     mode='lines', name='Particle trajectory', line=dict(color='black')
# ))
# fig.add_trace(go.Scatter3d(
#     x = [snapshots.states[0, 2, 0, 0] * scale],
#     y = [snapshots.states[0, 2, 0, 1] * scale],
#     z = [snapshots.states[0, 2, 0, 2] * scale],
#     mode='markers', name='Earlier position Particle', marker=dict(color='green')
# ))
# fig.add_trace(go.Scatter3d(
#     x = [snapshots.states[-1, 2, 0, 0] * scale],
#     y = [snapshots.states[-1, 2, 0, 1] * scale],
#     z = [snapshots.states[-1, 2, 0, 2] * scale],
#     mode='markers', name='Current position Particle', marker=dict(color='red')
# ))

# MW trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 0, 0, 0] * scale,
    y = snapshots.states[:, 0, 0, 1] * scale,
    z = snapshots.states[:, 0, 0, 2] * scale,
    mode='lines', name='MW trajectory', line=dict(color='purple')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 0, 0, 0] * scale],
    y = [snapshots.states[0, 0, 0, 1] * scale],
    z = [snapshots.states[0, 0, 0, 2] * scale],
    mode='markers', name='Current position MW', marker=dict(color='green')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 0, 0, 0] * scale],
    y = [snapshots.states[-1, 0, 0, 1] * scale],
    z = [snapshots.states[-1, 0, 0, 2] * scale],
    mode='markers', name='Earlier position MW', marker=dict(color='red')
))

# LMC trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 1, 0, 0] * scale,
    y = snapshots.states[:, 1, 0, 1] * scale,
    z = snapshots.states[:, 1, 0, 2] * scale,
    mode='lines', name='LMC trajectory', line=dict(color='blue')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 1, 0, 0] * scale],
    y = [snapshots.states[0, 1, 0, 1] * scale],
    z = [snapshots.states[0, 1, 0, 2] * scale],
    mode='markers', name='Current position LMC', marker=dict(color='green')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 1, 0, 0] * scale],
    y = [snapshots.states[-1, 1, 0, 1] * scale],
    z = [snapshots.states[-1, 1, 0, 2] * scale],
    mode='markers', name='Earlier position LMC', marker=dict(color='red')
))

fig.update_layout(scene=dict(
    xaxis_title = 'X [kpc]',
    yaxis_title = 'Y [kpc]',
    zaxis_title = 'Z [kpc]'
))

fig.write_html("./notebooks/reflex_motion_orbits.html")

energy_angular_momentum_plot(snapshots, code_units, "./notebooks/reflex_motion_conservation.png")