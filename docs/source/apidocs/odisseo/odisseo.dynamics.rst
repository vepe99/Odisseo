:py:mod:`odisseo.dynamics`
==========================

.. py:module:: odisseo.dynamics

.. autodoc2-docstring:: odisseo.dynamics
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`single_body_acc <odisseo.dynamics.single_body_acc>`
     - .. autodoc2-docstring:: odisseo.dynamics.single_body_acc
          :summary:
   * - :py:obj:`direct_acc <odisseo.dynamics.direct_acc>`
     - .. autodoc2-docstring:: odisseo.dynamics.direct_acc
          :summary:
   * - :py:obj:`direct_acc_laxmap <odisseo.dynamics.direct_acc_laxmap>`
     - .. autodoc2-docstring:: odisseo.dynamics.direct_acc_laxmap
          :summary:
   * - :py:obj:`direct_acc_matrix <odisseo.dynamics.direct_acc_matrix>`
     - .. autodoc2-docstring:: odisseo.dynamics.direct_acc_matrix
          :summary:
   * - :py:obj:`direct_acc_for_loop <odisseo.dynamics.direct_acc_for_loop>`
     - .. autodoc2-docstring:: odisseo.dynamics.direct_acc_for_loop
          :summary:
   * - :py:obj:`direct_acc_sharding <odisseo.dynamics.direct_acc_sharding>`
     - .. autodoc2-docstring:: odisseo.dynamics.direct_acc_sharding
          :summary:

API
~~~

.. py:function:: single_body_acc(particle_i: jax.numpy.ndarray, particle_j: jax.numpy.ndarray, mass_i: jax.numpy.ndarray, mass_j: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams) -> typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]
   :canonical: odisseo.dynamics.single_body_acc

   .. autodoc2-docstring:: odisseo.dynamics.single_body_acc

.. py:function:: direct_acc(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc

.. py:function:: direct_acc_laxmap(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc_laxmap

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_laxmap

.. py:function:: direct_acc_matrix(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential: bool = False) -> typing.Union[jax.numpy.ndarray, typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]
   :canonical: odisseo.dynamics.direct_acc_matrix

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_matrix

.. py:function:: direct_acc_for_loop(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential: bool = False) -> typing.Union[jax.numpy.ndarray, typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]
   :canonical: odisseo.dynamics.direct_acc_for_loop

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_for_loop

.. py:function:: direct_acc_sharding(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential: bool = False) -> typing.Union[jax.numpy.ndarray, typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]
   :canonical: odisseo.dynamics.direct_acc_sharding

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_sharding
