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

.. py:function:: single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params) -> jax.numpy.ndarray
   :canonical: odisseo.dynamics.single_body_acc

   .. autodoc2-docstring:: odisseo.dynamics.single_body_acc

.. py:function:: direct_acc(state, mass, config, params, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc

.. py:function:: direct_acc_laxmap(state, mass, config, params, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc_laxmap

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_laxmap

.. py:function:: direct_acc_matrix(state, mass, config, params, return_potential=False)
   :canonical: odisseo.dynamics.direct_acc_matrix

   .. autodoc2-docstring:: odisseo.dynamics.direct_acc_matrix
