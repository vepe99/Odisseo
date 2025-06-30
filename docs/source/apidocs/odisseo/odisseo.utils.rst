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
   * - :py:obj:`E_tot <odisseo.utils.E_tot>`
     - .. autodoc2-docstring:: odisseo.utils.E_tot
          :summary:
   * - :py:obj:`Angular_momentum <odisseo.utils.Angular_momentum>`
     - .. autodoc2-docstring:: odisseo.utils.Angular_momentum
          :summary:
   * - :py:obj:`halo_to_sun <odisseo.utils.halo_to_sun>`
     - .. autodoc2-docstring:: odisseo.utils.halo_to_sun
          :summary:
   * - :py:obj:`sun_to_gal <odisseo.utils.sun_to_gal>`
     - .. autodoc2-docstring:: odisseo.utils.sun_to_gal
          :summary:
   * - :py:obj:`gal_to_equat <odisseo.utils.gal_to_equat>`
     - .. autodoc2-docstring:: odisseo.utils.gal_to_equat
          :summary:
   * - :py:obj:`equat_to_gd1cart <odisseo.utils.equat_to_gd1cart>`
     - .. autodoc2-docstring:: odisseo.utils.equat_to_gd1cart
          :summary:
   * - :py:obj:`gd1cart_to_gd1 <odisseo.utils.gd1cart_to_gd1>`
     - .. autodoc2-docstring:: odisseo.utils.gd1cart_to_gd1
          :summary:
   * - :py:obj:`halo_to_gd1 <odisseo.utils.halo_to_gd1>`
     - .. autodoc2-docstring:: odisseo.utils.halo_to_gd1
          :summary:
   * - :py:obj:`equat_to_gd1 <odisseo.utils.equat_to_gd1>`
     - .. autodoc2-docstring:: odisseo.utils.equat_to_gd1
          :summary:
   * - :py:obj:`equat_to_gd1_velocity <odisseo.utils.equat_to_gd1_velocity>`
     - .. autodoc2-docstring:: odisseo.utils.equat_to_gd1_velocity
          :summary:
   * - :py:obj:`halo_to_gd1_velocity <odisseo.utils.halo_to_gd1_velocity>`
     - .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_velocity
          :summary:
   * - :py:obj:`halo_to_gd1_all <odisseo.utils.halo_to_gd1_all>`
     - .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_all
          :summary:
   * - :py:obj:`projection_on_GD1 <odisseo.utils.projection_on_GD1>`
     - .. autodoc2-docstring:: odisseo.utils.projection_on_GD1
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`jacobian_halo_to_gd1 <odisseo.utils.jacobian_halo_to_gd1>`
     - .. autodoc2-docstring:: odisseo.utils.jacobian_halo_to_gd1
          :summary:
   * - :py:obj:`halo_to_gd1_vmap <odisseo.utils.halo_to_gd1_vmap>`
     - .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_vmap
          :summary:
   * - :py:obj:`jacobian_equat_to_gd1 <odisseo.utils.jacobian_equat_to_gd1>`
     - .. autodoc2-docstring:: odisseo.utils.jacobian_equat_to_gd1
          :summary:
   * - :py:obj:`halo_to_gd1_velocity_vmap <odisseo.utils.halo_to_gd1_velocity_vmap>`
     - .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_velocity_vmap
          :summary:
   * - :py:obj:`gd1_projection_vmap <odisseo.utils.gd1_projection_vmap>`
     - .. autodoc2-docstring:: odisseo.utils.gd1_projection_vmap
          :summary:

API
~~~

.. py:function:: center_of_mass(state: jax.numpy.ndarray, mass: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.center_of_mass

   .. autodoc2-docstring:: odisseo.utils.center_of_mass

.. py:function:: E_kin(state: jax.numpy.ndarray, mass: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.E_kin

   .. autodoc2-docstring:: odisseo.utils.E_kin

.. py:function:: E_pot(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams) -> jax.numpy.ndarray
   :canonical: odisseo.utils.E_pot

   .. autodoc2-docstring:: odisseo.utils.E_pot

.. py:function:: E_tot(state: jax.numpy.ndarray, mass: jax.numpy.ndarray, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams) -> jax.numpy.ndarray
   :canonical: odisseo.utils.E_tot

   .. autodoc2-docstring:: odisseo.utils.E_tot

.. py:function:: Angular_momentum(state: jax.numpy.ndarray, mass: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.Angular_momentum

   .. autodoc2-docstring:: odisseo.utils.Angular_momentum

.. py:function:: halo_to_sun(Xhalo: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.halo_to_sun

   .. autodoc2-docstring:: odisseo.utils.halo_to_sun

.. py:function:: sun_to_gal(Xsun: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.sun_to_gal

   .. autodoc2-docstring:: odisseo.utils.sun_to_gal

.. py:function:: gal_to_equat(Xgal: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.gal_to_equat

   .. autodoc2-docstring:: odisseo.utils.gal_to_equat

.. py:function:: equat_to_gd1cart(Xequat: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.equat_to_gd1cart

   .. autodoc2-docstring:: odisseo.utils.equat_to_gd1cart

.. py:function:: gd1cart_to_gd1(Xgd1cart: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.gd1cart_to_gd1

   .. autodoc2-docstring:: odisseo.utils.gd1cart_to_gd1

.. py:function:: halo_to_gd1(Xhalo: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.halo_to_gd1

   .. autodoc2-docstring:: odisseo.utils.halo_to_gd1

.. py:data:: jacobian_halo_to_gd1
   :canonical: odisseo.utils.jacobian_halo_to_gd1
   :value: 'jit(...)'

   .. autodoc2-docstring:: odisseo.utils.jacobian_halo_to_gd1

.. py:data:: halo_to_gd1_vmap
   :canonical: odisseo.utils.halo_to_gd1_vmap
   :value: 'jit(...)'

   .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_vmap

.. py:function:: equat_to_gd1(Xequat: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.equat_to_gd1

   .. autodoc2-docstring:: odisseo.utils.equat_to_gd1

.. py:data:: jacobian_equat_to_gd1
   :canonical: odisseo.utils.jacobian_equat_to_gd1
   :value: 'jit(...)'

   .. autodoc2-docstring:: odisseo.utils.jacobian_equat_to_gd1

.. py:function:: equat_to_gd1_velocity(Xequat: jax.numpy.ndarray, Vequat: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.equat_to_gd1_velocity

   .. autodoc2-docstring:: odisseo.utils.equat_to_gd1_velocity

.. py:function:: halo_to_gd1_velocity(Xhalo: jax.numpy.ndarray, Vhalo: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.halo_to_gd1_velocity

   .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_velocity

.. py:data:: halo_to_gd1_velocity_vmap
   :canonical: odisseo.utils.halo_to_gd1_velocity_vmap
   :value: 'jit(...)'

   .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_velocity_vmap

.. py:function:: halo_to_gd1_all(Xhalo: jax.numpy.ndarray, Vhalo: jax.numpy.ndarray) -> jax.numpy.ndarray
   :canonical: odisseo.utils.halo_to_gd1_all

   .. autodoc2-docstring:: odisseo.utils.halo_to_gd1_all

.. py:data:: gd1_projection_vmap
   :canonical: odisseo.utils.gd1_projection_vmap
   :value: 'jit(...)'

   .. autodoc2-docstring:: odisseo.utils.gd1_projection_vmap

.. py:function:: projection_on_GD1(final_state, code_units: odisseo.units.CodeUnits) -> jax.numpy.ndarray
   :canonical: odisseo.utils.projection_on_GD1

   .. autodoc2-docstring:: odisseo.utils.projection_on_GD1
