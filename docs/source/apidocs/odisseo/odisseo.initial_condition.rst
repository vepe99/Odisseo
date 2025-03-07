:py:mod:`odisseo.initial_condition`
===================================

.. py:module:: odisseo.initial_condition

.. autodoc2-docstring:: odisseo.initial_condition
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Plummer_sphere <odisseo.initial_condition.Plummer_sphere>`
     - .. autodoc2-docstring:: odisseo.initial_condition.Plummer_sphere
          :summary:
   * - :py:obj:`Plummer_sphere_multiprocess <odisseo.initial_condition.Plummer_sphere_multiprocess>`
     - .. autodoc2-docstring:: odisseo.initial_condition.Plummer_sphere_multiprocess
          :summary:
   * - :py:obj:`ic_two_body <odisseo.initial_condition.ic_two_body>`
     - .. autodoc2-docstring:: odisseo.initial_condition.ic_two_body
          :summary:
   * - :py:obj:`sample_position_on_sphere <odisseo.initial_condition.sample_position_on_sphere>`
     - .. autodoc2-docstring:: odisseo.initial_condition.sample_position_on_sphere
          :summary:
   * - :py:obj:`inclined_circular_velocity <odisseo.initial_condition.inclined_circular_velocity>`
     - .. autodoc2-docstring:: odisseo.initial_condition.inclined_circular_velocity
          :summary:

API
~~~

.. py:function:: Plummer_sphere(key, config, params)
   :canonical: odisseo.initial_condition.Plummer_sphere

   .. autodoc2-docstring:: odisseo.initial_condition.Plummer_sphere

.. py:function:: Plummer_sphere_multiprocess(mass, config, params)
   :canonical: odisseo.initial_condition.Plummer_sphere_multiprocess

   .. autodoc2-docstring:: odisseo.initial_condition.Plummer_sphere_multiprocess

.. py:function:: ic_two_body(mass1: float, mass2: float, rp: float, e: float, config, params)
   :canonical: odisseo.initial_condition.ic_two_body

   .. autodoc2-docstring:: odisseo.initial_condition.ic_two_body

.. py:function:: sample_position_on_sphere(key, r_p, num_samples=1)
   :canonical: odisseo.initial_condition.sample_position_on_sphere

   .. autodoc2-docstring:: odisseo.initial_condition.sample_position_on_sphere

.. py:function:: inclined_circular_velocity(position, v_c, inclination)
   :canonical: odisseo.initial_condition.inclined_circular_velocity

   .. autodoc2-docstring:: odisseo.initial_condition.inclined_circular_velocity
