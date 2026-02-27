print("Running reflex motion test")

from autocvd import autocvd
autocvd(num_gpus = 1)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax.numpy as jnp
from jax import random

import plotly.graph_objects as go

from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import NFW_POTENTIAL, DIFFRAX_BACKEND
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import energy_angular_momentum_plot

print("Imports successful")

# Setup
G = 1
code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
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
                          external_accelerations = (NFW_POTENTIAL,), 
                          reflex_motion = True
)       

params = SimulationParams(t_end = 1 * u.Gyr.to(code_units.code_time),
                          G = code_units.G
) 

key = random.PRNGKey(1)

# State of test particle
pos = jnp.array([[10., 0., 0.]]) * u.kpc.to(code_units.code_length)
vel = jnp.array([[0., 100., 0.]]) * (u.km/u.s).to(code_units.code_velocity)
mass = jnp.array([1e4]) * u.Msun.to(code_units.code_mass)
state = construct_initial_state(pos, vel) # state is a (N_particles x 2 x 3)

# Add the Milky Way as particle in index 0
MW_pos = jnp.array([[0., 0., 0.]]) * u.kpc.to(code_units.code_length)
MW_vel = jnp.array([[0., 0., 10.]]) * (u.km/u.s).to(code_units.code_velocity)
MW_mass = jnp.array([0.]) * u.Msun.to(code_units.code_mass) # Set its mass to zero, as its effect is included as an external potential
final_state = jnp.concatenate([construct_initial_state(MW_pos, MW_vel), state], axis=0)
final_mass = jnp.concatenate([MW_mass, mass], axis=0) 

print("Setup successful")

# Simulation
snapshots = time_integration(final_state, final_mass, config, params) # snapshots is a NameTuple with states=(N_snapshots x N_particles x 2 x 3) array

print("Time integration successful")

# Visualization
scale = code_units.code_length.to(u.kpc)

fig = go.Figure()

# Particle trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 1, 0, 0] * scale,
    y = snapshots.states[:, 1, 0, 1] * scale,
    z = snapshots.states[:, 1, 0, 2] * scale,
    mode='lines', name='Particle trajectory', line=dict(color='black')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 1, 0, 0] * scale],
    y = [snapshots.states[0, 1, 0, 1] * scale],
    z = [snapshots.states[0, 1, 0, 2] * scale],
    mode='markers', name='Initial position Particle', marker=dict(color='green')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 1, 0, 0] * scale],
    y = [snapshots.states[-1, 1, 0, 1] * scale],
    z = [snapshots.states[-1, 1, 0, 2] * scale],
    mode='markers', name='Final position Particle', marker=dict(color='red')
))

# MW trajectory
fig.add_trace(go.Scatter3d(
    x = snapshots.states[:, 0, 0, 0] * scale,
    y = snapshots.states[:, 0, 0, 1] * scale,
    z = snapshots.states[:, 0, 0, 2] * scale,
    mode='lines', name='MW trajectory', line=dict(color='gray')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[0, 0, 0, 0] * scale],
    y = [snapshots.states[0, 0, 0, 1] * scale],
    z = [snapshots.states[0, 0, 0, 2] * scale],
    mode='markers', name='Initial position MW', marker=dict(color='blue')
))
fig.add_trace(go.Scatter3d(
    x = [snapshots.states[-1, 0, 0, 0] * scale],
    y = [snapshots.states[-1, 0, 0, 1] * scale],
    z = [snapshots.states[-1, 0, 0, 2] * scale],
    mode='markers', name='Final position MW', marker=dict(color='yellow')
))

fig.update_layout(scene=dict(
    xaxis_title = 'X [kpc]',
    yaxis_title = 'Y [kpc]',
    zaxis_title = 'Z [kpc]'
))

fig.write_html("./tests/reflex_motion_3d_orbit.html")

energy_angular_momentum_plot(snapshots, code_units, "./tests/reflex_motion_energy_angular_momentum.png")