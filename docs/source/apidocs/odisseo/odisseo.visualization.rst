:py:mod:`odisseo.visualization`
===============================

.. py:module:: odisseo.visualization

.. autodoc2-docstring:: odisseo.visualization
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`energy_angular_momentum_plot <odisseo.visualization.energy_angular_momentum_plot>`
     - .. autodoc2-docstring:: odisseo.visualization.energy_angular_momentum_plot
          :summary:
   * - :py:obj:`plot_last_snapshot <odisseo.visualization.plot_last_snapshot>`
     - .. autodoc2-docstring:: odisseo.visualization.plot_last_snapshot
          :summary:
   * - :py:obj:`plot_orbit <odisseo.visualization.plot_orbit>`
     - .. autodoc2-docstring:: odisseo.visualization.plot_orbit
          :summary:
   * - :py:obj:`plot_sky_projection <odisseo.visualization.plot_sky_projection>`
     - .. autodoc2-docstring:: odisseo.visualization.plot_sky_projection
          :summary:
   * - :py:obj:`create_3d_gif <odisseo.visualization.create_3d_gif>`
     - .. autodoc2-docstring:: odisseo.visualization.create_3d_gif
          :summary:
   * - :py:obj:`create_3d_gif_velocitycoding <odisseo.visualization.create_3d_gif_velocitycoding>`
     - .. autodoc2-docstring:: odisseo.visualization.create_3d_gif_velocitycoding
          :summary:
   * - :py:obj:`create_projection_gif <odisseo.visualization.create_projection_gif>`
     - .. autodoc2-docstring:: odisseo.visualization.create_projection_gif
          :summary:
   * - :py:obj:`build_spiral_stars <odisseo.visualization.build_spiral_stars>`
     - .. autodoc2-docstring:: odisseo.visualization.build_spiral_stars
          :summary:
   * - :py:obj:`build_spiral_arms <odisseo.visualization.build_spiral_arms>`
     - .. autodoc2-docstring:: odisseo.visualization.build_spiral_arms
          :summary:
   * - :py:obj:`spherical_coords <odisseo.visualization.spherical_coords>`
     - .. autodoc2-docstring:: odisseo.visualization.spherical_coords
          :summary:
   * - :py:obj:`build_core_stars <odisseo.visualization.build_core_stars>`
     - .. autodoc2-docstring:: odisseo.visualization.build_core_stars
          :summary:
   * - :py:obj:`haze <odisseo.visualization.haze>`
     - .. autodoc2-docstring:: odisseo.visualization.haze
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SCALE <odisseo.visualization.SCALE>`
     - .. autodoc2-docstring:: odisseo.visualization.SCALE
          :summary:
   * - :py:obj:`arms_info <odisseo.visualization.arms_info>`
     - .. autodoc2-docstring:: odisseo.visualization.arms_info
          :summary:

API
~~~

.. py:function:: energy_angular_momentum_plot(snapshots, code_units, filename=None)
   :canonical: odisseo.visualization.energy_angular_momentum_plot

   .. autodoc2-docstring:: odisseo.visualization.energy_angular_momentum_plot

.. py:function:: plot_last_snapshot(snapshots, code_units, rp, plotting_units_length, filename=None)
   :canonical: odisseo.visualization.plot_last_snapshot

   .. autodoc2-docstring:: odisseo.visualization.plot_last_snapshot

.. py:function:: plot_orbit(snapshots, ax_lim, code_units, plotting_units_length, config, filename=None)
   :canonical: odisseo.visualization.plot_orbit

   .. autodoc2-docstring:: odisseo.visualization.plot_orbit

.. py:function:: plot_sky_projection(snapshots, code_units, plotting_units_length, filename=None)
   :canonical: odisseo.visualization.plot_sky_projection

   .. autodoc2-docstring:: odisseo.visualization.plot_sky_projection

.. py:function:: create_3d_gif(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, filename=None)
   :canonical: odisseo.visualization.create_3d_gif

   .. autodoc2-docstring:: odisseo.visualization.create_3d_gif

.. py:function:: create_3d_gif_velocitycoding(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, vmin=None, vmax=None, filename=None)
   :canonical: odisseo.visualization.create_3d_gif_velocitycoding

   .. autodoc2-docstring:: odisseo.visualization.create_3d_gif_velocitycoding

.. py:function:: create_projection_gif(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, filename=None)
   :canonical: odisseo.visualization.create_projection_gif

   .. autodoc2-docstring:: odisseo.visualization.create_projection_gif

.. py:data:: SCALE
   :canonical: odisseo.visualization.SCALE
   :value: 26

   .. autodoc2-docstring:: odisseo.visualization.SCALE

.. py:function:: build_spiral_stars(b, r, rot_fac, fuz_fac)
   :canonical: odisseo.visualization.build_spiral_stars

   .. autodoc2-docstring:: odisseo.visualization.build_spiral_stars

.. py:data:: arms_info
   :canonical: odisseo.visualization.arms_info
   :value: [(), (), (), (), (), (), (), ()]

   .. autodoc2-docstring:: odisseo.visualization.arms_info

.. py:function:: build_spiral_arms(b, arms_info)
   :canonical: odisseo.visualization.build_spiral_arms

   .. autodoc2-docstring:: odisseo.visualization.build_spiral_arms

.. py:function:: spherical_coords(num_pts, radius)
   :canonical: odisseo.visualization.spherical_coords

   .. autodoc2-docstring:: odisseo.visualization.spherical_coords

.. py:function:: build_core_stars(scale_factor)
   :canonical: odisseo.visualization.build_core_stars

   .. autodoc2-docstring:: odisseo.visualization.build_core_stars

.. py:function:: haze(scale_factor, r_mult, z_mult, density)
   :canonical: odisseo.visualization.haze

   .. autodoc2-docstring:: odisseo.visualization.haze
