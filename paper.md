---
title: 'Odisseo: A Differentiable N-body Code for Gradient-Informed Galactic Dynamics'
tags:
  - Python
  - jax
  - astronomy
  - astrophysics
  - galactic dynamics
  - stellar streams
  - N-body
  - simulation-based inference
authors:
  - name: Giuseppe Viterbo
    orcid: 0009-0001-1864-8476
    affiliation: "1"
  - name: Tobias Buck
    orcid: 0000-0003-2027-399X
    affilitation: "1"

affiliations:
 - name: "IWR, Heidelberg University"
   index: 1
date: 01 July 2025
bibliography: paper.bib
---

# Background

Stellar streams are the fossilized remnants of dwarf galaxies and globular clusters that have been tidally disrupted by the gravitational potential of their host galaxy. These structures, observed as coherent filaments of stars in the Milky Way and other nearby galaxies, are powerful probes of astrophysics. Their morphology and kinematics encode detailed information about the host galaxy's gravitational field, making them ideal for mapping its shape. Furthermore, studying the properties of streams and their progenitors is key to unraveling the hierarchical assembly history of galaxies.

The primary theoretical tool for studying stream dynamics is the N-body simulation, which models the gravitational interactions of a system of particles. For decades, the standard workflow has involved running computationally expensive simulations, comparing their projected outputs to observational data via summary statistics, and then using statistical methods like MCMC to explore the vast parameter spaces. While successful, this approaches loses information by compressing rich datasets into simple statistics, and struggles with the high-dimensional parameter spaces required by increasingly complex models. 

# Statement of Need

Inspired by the work of [`alvey:2024`] and [`nibauer:2024`] on stellar stream differentiable simulators, with Odisseo (Optimized Differentiable Integrator for Stellar Systems Evolution of Orbit) we intend to offer a general purpose, highly modular, full N-body package that can be use for detail inference pipeline by taking advantage of the full information present in the phase-space. To fully exploit the wealth of information in upcoming astronomical surveys, the field requires new tools that can overcome the limitations of traditional methods. As demonstrated by recent developments, a promising path forward lies in leveraging differentiable programming and modern simulation-based inference (SBI) techniques [`@holzschuh:2024`].

By providing a fully differentiable N-body simulator built on JAX, Odisseo directly addresses the key bottlenecks of the previous paradigm. Its differentiability allows for the direct use of simulation gradients to guide parameter inference, enabling a move from inefficient parameter searches to highly efficient, gradient-informed methods. 
This approach offers two major advantages. First, it allows for direct optimization via gradient descent, making it possible to jointly and efficiently infer parameters of both the progenitor system and the host galaxy's potential. Second, it is a key enabler for advanced statistical methods like gradient-enhanced SBI, which combine the power of Bayesian inference with the efficiency of gradients to tackle otherwise intractable, high-dimensional problems. 

# Odisseo Overview

Odisseo is a Python package written in a purely functional style to integrate seamlessly with the JAX ecosystem. Its design philosophy is to provide a simple, flexible, and powerful tool for inference-focused N-body simulations. Key features include:

*   **Differentiable N-body**: The core of the package is a differentiable N-body code. The final state of the simulation is differentiable with respect to its initial parameters, including initial particle positions, velocities, and progenitor mass.

*   **Modularity**: The modularity of the code allows to swap each component, allowing for quick prototyping and model testing. 

*   **JAX Native**: The code is built entirely on JAX, enabling end-to-end JIT compilation for high performance, automatic vectorization (`vmap`) for trivially parallelizing computations, and automatic differentiation (`grad`) for gradient calculations.

*   **External Gravitational Potentials**: Odisseo allows for the inclusion of arbitrary external potentials to represent the host galaxy. This is essential for realistically modeling the tidal disruption of satellite systems in a Milky Way-like environment.

* **Unit Conversion**: The conversion between physical and simulation units is handled with a simple `CodeUnits` class that wrapps around `astropy` functionality [`astropy:2022`].

*   **Gradient-Informed Inference**: The package is explicitly designed for parameter inference. By defining a differentiable loss function that compares simulation outputs to data, users can leverage the computed gradients in optimizers for gradient descent, use them to augment the learning process in SBI frameworks, or enhance the output of already existing posterior results with differentiable posterior predictive checks. 


# Running a simulation

To run a simulation we need 4 main components :
* **configuration**: changing them would require recompiling the simulation for `jax.jit`, because it would causes changes in shapes of the arrays.
* **parameters**: physical paramters with respect to we can take the `jax.grad` of a scalar function.
* **initial_conditions**: the initial state of the simulation, it contains the `positions` and `velocity` of all the particles. The `masses` are a separate array with the same shape of `initial_conditions`.
* **time_integration**: main function that evolves the particles state.

In the following use case we show how to evolve a Plummer sphere in a Milky-Way like potential to replicate a stellar stream with known progenitor position and velocity

```python
from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import MNParams, NFWParams, PlummerParams, PSPParams
from odisseo.option_classes import MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits

# Code units conversion
code_length = 10.0 * u.kpc
code_mass = 1e8 * u.Msun
G = 1 
code_units = CodeUnits(code_length, code_mass, G=G)

# Define the configuration and parameters for the particles
config = SimulationConfig(N_particles=int(1_000), 
                          return_snapshots=True, 
                          num_snapshots=100, 
                          num_timesteps=1000, 
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL ), 
                          acceleration_scheme=DIRECT_ACC_MATRIX,
                          softening=(0.1 * u.pc).to(code_units.code_length).value)                                #default values

params = SimulationParams(t_end = (10 * u.Gyr).to(code_units.code_time).value,  
                          Plummer_params= PlummerParams(Mtot=(1e8 * u.Msun).to(code_units.code_mass).value,        #Plummer sphere parameters
                                                        a=(1 * u.kpc).to(code_units.code_length).value),
                          NFW_params= NFWParams(Mvir=(4.3683325e11 * u.Msun).to(code_units.code_mass).value,        #Navarro-Frank-White halo model parameters
                                               r_s= (16.0 * u.kpc).to(code_units.code_length).value,),      
                          MN_params= MNParams(M = (68_193_902_782.346756 * u.Msun).to(code_units.code_mass).value,  #Miamoto-Nagai disk model parameters
                                              a = (3.0 * u.kpc).to(code_units.code_length).value,
                                              b = (0.280 * u.kpc).to(code_units.code_length).value),
                          PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),             #PowerSphericalPotentialwCutoff bulge model 
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(code_units.code_length).value),  
                          G=G, ) 

# Create the TARGET stream for the GD-1 stream
key = random.PRNGKey(43)
#set up the particles in the initial state
positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)
#the center of mass needs to be integrated backwards in time first 
config_com = config._replace(N_particles=1,)
params_com = params._replace(t_end=-params.t_end,)

#this is the final position of the cluster, we need to integrate backwards in time 
pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)
mass_com = jnp.array([params.Plummer_params.Mtot]) 

#we construmt the initial state of the com 
initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
#we run the simulation backwards in time for the center of mass
final_state_com = time_integration(primitive_state = initial_state_com, mass = mass_com, config=config_com, params=params_com)
#we calculate the final position and velocity of the center of mass
pos_com = final_state_com[:, 0]
vel_com = final_state_com[:, 1]
#we construct the initial state of the Plummer sphere
positions, velocities, mass = Plummer_sphere(key=random.PRNGKey(rng_key), params=params, config=config)
#we add the center of mass position and velocity to the Plummer sphere particles
positions = positions + pos_com
velocities = velocities + vel_com
#initialize the initial state
initial_state_stream = construct_initial_state(positions = positions, velocities = velocities, )
#run the simulation
final_state = time_integration(primitive_state =initial_state_stream, mass = mass, config = config, params = params)
```


# Acknowledgements

We acknowledge [funding source] for their support of this project. We also thank [names of contributors] for their helpful feedback and contributions.

# References