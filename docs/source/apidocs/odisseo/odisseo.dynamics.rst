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

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DIRECT_ACC <odisseo.dynamics.DIRECT_ACC>`
     - .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC
          :summary:
   * - :py:obj:`DIRECT_ACC_LAXMAP <odisseo.dynamics.DIRECT_ACC_LAXMAP>`
     - .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_LAXMAP
          :summary:
   * - :py:obj:`DIRECT_ACC_MATRIX <odisseo.dynamics.DIRECT_ACC_MATRIX>`
     - .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_MATRIX
          :summary:
   * - :py:obj:`DIRECT_ACC_FOR_LOOP <odisseo.dynamics.DIRECT_ACC_FOR_LOOP>`
     - .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_FOR_LOOP
          :summary:
   * - :py:obj:`DIRECT_ACC_SHARDING <odisseo.dynamics.DIRECT_ACC_SHARDING>`
     - .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_SHARDING
          :summary:

API
~~~

.. py:data:: DIRECT_ACC
   :canonical: odisseo.dynamics.DIRECT_ACC
   :value: 0

   .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC

.. py:data:: DIRECT_ACC_LAXMAP
   :canonical: odisseo.dynamics.DIRECT_ACC_LAXMAP
   :value: 1

   .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_LAXMAP

.. py:data:: DIRECT_ACC_MATRIX
   :canonical: odisseo.dynamics.DIRECT_ACC_MATRIX
   :value: 2

   .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_MATRIX

.. py:data:: DIRECT_ACC_FOR_LOOP
   :canonical: odisseo.dynamics.DIRECT_ACC_FOR_LOOP
   :value: 3

   .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_FOR_LOOP

.. py:data:: DIRECT_ACC_SHARDING
   :canonical: odisseo.dynamics.DIRECT_ACC_SHARDING
   :value: 4

   .. autodoc2-docstring:: odisseo.dynamics.DIRECT_ACC_SHARDING

.. py:function:: single_body_acc(particle_i: jax.numpy.ndarray, particle_j: jax.numpy.ndarray, mass_i: float, mass_j: float, config: typing.NamedTuple, params: typing.NamedTuple) -> jax.numpy.ndarray
   :canonical: odisseo.dynamics.single_body_acc

   .. autodoc2-docstring:: odisseo.dynamics.single_body_acc

.. py:function:: direct_acc(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: typing.NamedTuple, params: typing.NamedTuple, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc

.. py:function:: direct_acc_laxmap(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: typing.NamedTuple, params: typing.NamedTuple, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc_laxmap

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_laxmap

.. py:function:: direct_acc_matrix(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: typing.NamedTuple, params: typing.NamedTuple, return_potential: bool = False) -> typing.Union[jax.numpy.ndarray, typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]
   :canonical: odisseo.dynamics.direct_acc_matrix

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_matrix

.. py:function:: direct_acc_for_loop(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: typing.NamedTuple, params: typing.NamedTuple, return_potential: bool = False) -> typing.Union[jax.numpy.ndarray, typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]
   :canonical: odisseo.dynamics.direct_acc_for_loop

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_for_loop

.. py:function:: direct_acc_sharding(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: typing.NamedTuple, params: typing.NamedTuple, return_potential: bool = False) -> typing.Union[jax.numpy.ndarray, typing.Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]
   :canonical: odisseo.dynamics.direct_acc_sharding

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_sharding
