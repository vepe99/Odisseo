:py:mod:`odisseo.utils`
=======================

.. py:module:: odisseo.utils

.. autodoc2-docstring:: odisseo.utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`center_of_mass <odisseo.utils.center_of_mass>`
     - .. autodoc2-docstring:: odisseo.utils.center_of_mass
          :summary:
   * - :py:obj:`E_kin <odisseo.utils.E_kin>`
     - .. autodoc2-docstring:: odisseo.utils.E_kin
          :summary:
   * - :py:obj:`E_pot <odisseo.utils.E_pot>`
     - .. autodoc2-docstring:: odisseo.utils.E_pot
          :summary:
   * - :py:obj:`Angular_momentum <odisseo.utils.Angular_momentum>`
     - .. autodoc2-docstring:: odisseo.utils.Angular_momentum
          :summary:

API
~~~

.. py:function:: center_of_mass(state: jax.numpy.ndarray, mass: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.center_of_mass

   .. autodoc2-docstring:: odisseo.utils.center_of_mass

.. py:function:: E_kin(state: jax.numpy.ndarray, mass: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.E_kin

   .. autodoc2-docstring:: odisseo.utils.E_kin

.. py:function:: E_pot(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: typing.NamedTuple, params: typing.NamedTuple)
   :canonical: odisseo.utils.E_pot

   .. autodoc2-docstring:: odisseo.utils.E_pot

.. py:function:: Angular_momentum(state: jax.numpy.ndarray, mass: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.Angular_momentum

   .. autodoc2-docstring:: odisseo.utils.Angular_momentum
