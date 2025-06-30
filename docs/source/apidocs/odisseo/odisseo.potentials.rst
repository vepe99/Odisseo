:py:mod:`odisseo.potentials`
============================

.. py:module:: odisseo.potentials

.. autodoc2-docstring:: odisseo.potentials
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`combined_external_acceleration <odisseo.potentials.combined_external_acceleration>`
     - .. autodoc2-docstring:: odisseo.potentials.combined_external_acceleration
          :summary:
   * - :py:obj:`combined_external_acceleration_vmpa_switch <odisseo.potentials.combined_external_acceleration_vmpa_switch>`
     - .. autodoc2-docstring:: odisseo.potentials.combined_external_acceleration_vmpa_switch
          :summary:
   * - :py:obj:`NFW <odisseo.potentials.NFW>`
     - .. autodoc2-docstring:: odisseo.potentials.NFW
          :summary:
   * - :py:obj:`point_mass <odisseo.potentials.point_mass>`
     - .. autodoc2-docstring:: odisseo.potentials.point_mass
          :summary:
   * - :py:obj:`MyamotoNagai <odisseo.potentials.MyamotoNagai>`
     - .. autodoc2-docstring:: odisseo.potentials.MyamotoNagai
          :summary:
   * - :py:obj:`PowerSphericalPotentialwCutoff <odisseo.potentials.PowerSphericalPotentialwCutoff>`
     - .. autodoc2-docstring:: odisseo.potentials.PowerSphericalPotentialwCutoff
          :summary:
   * - :py:obj:`logarithmic_potential <odisseo.potentials.logarithmic_potential>`
     - .. autodoc2-docstring:: odisseo.potentials.logarithmic_potential
          :summary:

API
~~~

.. py:function:: combined_external_acceleration(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.combined_external_acceleration

   .. autodoc2-docstring:: odisseo.potentials.combined_external_acceleration

.. py:function:: combined_external_acceleration_vmpa_switch(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.combined_external_acceleration_vmpa_switch

   .. autodoc2-docstring:: odisseo.potentials.combined_external_acceleration_vmpa_switch

.. py:function:: NFW(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.NFW

   .. autodoc2-docstring:: odisseo.potentials.NFW

.. py:function:: point_mass(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.point_mass

   .. autodoc2-docstring:: odisseo.potentials.point_mass

.. py:function:: MyamotoNagai(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.MyamotoNagai

   .. autodoc2-docstring:: odisseo.potentials.MyamotoNagai

.. py:function:: PowerSphericalPotentialwCutoff(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.PowerSphericalPotentialwCutoff

   .. autodoc2-docstring:: odisseo.potentials.PowerSphericalPotentialwCutoff

.. py:function:: logarithmic_potential(state: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams, return_potential=False)
   :canonical: odisseo.potentials.logarithmic_potential

   .. autodoc2-docstring:: odisseo.potentials.logarithmic_potential
