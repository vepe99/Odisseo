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
date: 21 July 2025
bibliography: paper.bib
---

# Background
N-body simulations, which model the interactions within a system of particles, are a fundamental tool in computational physics with widespread applications [e.g. planetary science, stellar cluster dynamics, cosmology, molecular dynamics]. While Odisseo (Optimized Differentiable Integrator for Stellar Systems Evolution of Orbit) can be used in any context where an N-body simulation is useful, a key motivating application in galactic astrophysics is the study of stellar streams. Stellar streams are the fossilized remnants of dwarf galaxies and globular clusters that have been tidally disrupted by the gravitational potential of their host galaxy. These structures, observed as coherent filaments of stars in the Milky Way and other nearby galaxies, are powerful probes of astrophysics. Their morphology and kinematics encode detailed information about the host galaxy's gravitational field, making them ideal for mapping its shape. Furthermore, studying the properties of streams and their progenitors is key to unraveling the hierarchical assembly history of galaxies.

The standard workflow has involved running computationally expensive simulations, comparing their projected outputs to observational data via summary statistics, and then using statistical methods like MCMC to explore the vast parameter spaces. While successful, this approaches loses information by compressing rich datasets into simple statistics, and struggles with the high-dimensional parameter spaces required by increasingly complex models. With Odisseo we aim to explore how the new paradigm of differentiable simulations and automatic-differentiation can be adopted to challenge many body problems. 


# Statement of Need

Inspired by the work of [@alvey:2024] and [@nibauer:2024] on stellar stream differentiable simulators, with Odisseo we intend to offer a general purpose, highly modular, full N-body simulator package that can be use for detail inference pipeline by taking advantage of the full information present in the phase-space. The main goal is to explore the joint posterior distribution of progenitor and external potential parameters in the context of galactic dynamics. As demonstrated by recent developments, a promising path for inverse modeling techniques lies in leveraging differentiable programming and modern simulation-based inference (SBI) techniques [@holzschuh:2024].

By providing a fully differentiable N-body simulator built on JAX, Odisseo directly addresses the key bottlenecks of the standard inference pipeline (MCMC). Its differentiability allows for the direct use of simulation gradients to guide parameter inference, enabling a move from inefficient parameter searches to highly efficient, gradient-informed methods. 

Odisseo is designed with open-source, community-driven development in mind, providing a robust and accessible foundation that can be extended with new physics models and numerical methods.

# Odisseo Overview

Odisseo is a Python package written in a purely functional style to integrate seamlessly with the JAX ecosystem. Its design philosophy is to provide a simple, flexible, and powerful tool for inference-focused N-body simulations. Key features include:

*   **End-to-End Differentiable**: The entire simulation pipeline is differentiable. The final state of the particles is differentiable with respect to the initial parameters, including initial conditions, total time of integration, particle masses, and parameters of the external potentials.

*   **Modularity and Extensibility**: The code is highly modular. The functional design allows for individual components —such as integrators, external potentials, or initial condition generators— to be easily swapped or extended by the user. This facilitates rapid prototyping and model testing.

*   **JAX Native and Cross-Platform**: Built entirely on JAX [@jax:2018], Odisseo enables end-to-end JIT compilation for high performance, automatic vectorization (`jax.vmap`) for trivial parallelization, and automatic differentiation (`jax.grad`). This ensures high performance across diverse hardware, including CPUs, GPUs, and TPUs.

*   **External Potentials**: Odisseo allows for the inclusion of arbitrary external potentials. This is essential for realistically modeling the tidal disruption of satellite systems in a Milky Way-like environment. It can be trivially generalized to physical settings where external potential are important (e.g. molecular dynamics)

*   **Unit Conversion**: The conversion between physical and simulation units is handled with a simple `CodeUnits` class that wraps around `astropy` functionality [@astropy:2022].


# Running a simulation

Four main components are needed to run a simulation :

*   **configuration**: it handles the shapes of the arrays in the simulations (e.g. number of particles, number of time steps) and all the components for which recompilation would be required if changed.

*   **parameters**: it contains the physical parameters with respect to we can differentiate through the time stepping.

*   **initial_conditions**: the initial state of the simulation, it contains the `positions` and `velocity` of all the particles. The `masses` are a separate array with the same length of `initial_conditions`.

*   **time_integration**: the main function that perform the evolution of the particles state.

Examples on how to set up different problems are presented in the [documentation](https://odisseo.readthedocs.io/en/latest/).

# Acknowledgements

We acknowledge [funding source] for their support of this project. We also thank [names of contributors] for their helpful feedback and contributions.

# References
