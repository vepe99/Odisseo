:py:mod:`odisseo.integrators`
=============================

.. py:module:: odisseo.integrators

.. autodoc2-docstring:: odisseo.integrators
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`leapfrog <odisseo.integrators.leapfrog>`
     - .. autodoc2-docstring:: odisseo.integrators.leapfrog
          :summary:
   * - :py:obj:`RungeKutta4 <odisseo.integrators.RungeKutta4>`
     - .. autodoc2-docstring:: odisseo.integrators.RungeKutta4
          :summary:
   * - :py:obj:`diffrax_solver <odisseo.integrators.diffrax_solver>`
     - .. autodoc2-docstring:: odisseo.integrators.diffrax_solver
          :summary:

API
~~~

.. py:function:: leapfrog(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, dt: jaxtyping.Scalar, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams)
   :canonical: odisseo.integrators.leapfrog

   .. autodoc2-docstring:: odisseo.integrators.leapfrog

.. py:function:: RungeKutta4(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, dt: jaxtyping.Scalar, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams)
   :canonical: odisseo.integrators.RungeKutta4

   .. autodoc2-docstring:: odisseo.integrators.RungeKutta4

.. py:function:: diffrax_solver(state, mass: jax.numpy.ndarray, dt: jaxtyping.Scalar, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams) -> jax.numpy.ndarray
   :canonical: odisseo.integrators.diffrax_solver

   .. autodoc2-docstring:: odisseo.integrators.diffrax_solver
