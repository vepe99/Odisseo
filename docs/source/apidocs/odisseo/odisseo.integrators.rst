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

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LEAPFROG <odisseo.integrators.LEAPFROG>`
     - .. autodoc2-docstring:: odisseo.integrators.LEAPFROG
          :summary:
   * - :py:obj:`RK4 <odisseo.integrators.RK4>`
     - .. autodoc2-docstring:: odisseo.integrators.RK4
          :summary:

API
~~~

.. py:data:: LEAPFROG
   :canonical: odisseo.integrators.LEAPFROG
   :value: 0

   .. autodoc2-docstring:: odisseo.integrators.LEAPFROG

.. py:data:: RK4
   :canonical: odisseo.integrators.RK4
   :value: 1

   .. autodoc2-docstring:: odisseo.integrators.RK4

.. py:function:: leapfrog(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, dt: jaxtyping.Float, config: typing.NamedTuple, params: typing.NamedTuple)
   :canonical: odisseo.integrators.leapfrog

   .. autodoc2-docstring:: odisseo.integrators.leapfrog

.. py:function:: RungeKutta4(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, dt: jaxtyping.Float, config: typing.NamedTuple, params: typing.NamedTuple)
   :canonical: odisseo.integrators.RungeKutta4

   .. autodoc2-docstring:: odisseo.integrators.RungeKutta4
