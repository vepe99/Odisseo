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

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NFW_POTENTIAL <odisseo.potentials.NFW_POTENTIAL>`
     - .. autodoc2-docstring:: odisseo.potentials.NFW_POTENTIAL
          :summary:
   * - :py:obj:`POINT_MASS <odisseo.potentials.POINT_MASS>`
     - .. autodoc2-docstring:: odisseo.potentials.POINT_MASS
          :summary:
   * - :py:obj:`MN_POTENTIAL <odisseo.potentials.MN_POTENTIAL>`
     - .. autodoc2-docstring:: odisseo.potentials.MN_POTENTIAL
          :summary:

API
~~~

.. py:data:: NFW_POTENTIAL
   :canonical: odisseo.potentials.NFW_POTENTIAL
   :value: 0

   .. autodoc2-docstring:: odisseo.potentials.NFW_POTENTIAL

.. py:data:: POINT_MASS
   :canonical: odisseo.potentials.POINT_MASS
   :value: 1

   .. autodoc2-docstring:: odisseo.potentials.POINT_MASS

.. py:data:: MN_POTENTIAL
   :canonical: odisseo.potentials.MN_POTENTIAL
   :value: 2

   .. autodoc2-docstring:: odisseo.potentials.MN_POTENTIAL

.. py:function:: combined_external_acceleration(state, config, params, return_potential=False)
   :canonical: odisseo.potentials.combined_external_acceleration

   .. autodoc2-docstring:: odisseo.potentials.combined_external_acceleration

.. py:function:: combined_external_acceleration_vmpa_switch(state, config, params, return_potential=False)
   :canonical: odisseo.potentials.combined_external_acceleration_vmpa_switch

   .. autodoc2-docstring:: odisseo.potentials.combined_external_acceleration_vmpa_switch

.. py:function:: NFW(state, config, params, return_potential=False)
   :canonical: odisseo.potentials.NFW

   .. autodoc2-docstring:: odisseo.potentials.NFW

.. py:function:: point_mass(state, config, params, return_potential=False)
   :canonical: odisseo.potentials.point_mass

   .. autodoc2-docstring:: odisseo.potentials.point_mass

.. py:function:: MyamotoNagai(state, config, params, return_potential=False)
   :canonical: odisseo.potentials.MyamotoNagai

   .. autodoc2-docstring:: odisseo.potentials.MyamotoNagai
