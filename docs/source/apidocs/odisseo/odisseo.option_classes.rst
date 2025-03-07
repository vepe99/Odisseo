:py:mod:`odisseo.option_classes`
================================

.. py:module:: odisseo.option_classes

.. autodoc2-docstring:: odisseo.option_classes
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PlummerParams <odisseo.option_classes.PlummerParams>`
     - .. autodoc2-docstring:: odisseo.option_classes.PlummerParams
          :summary:
   * - :py:obj:`NFWParams <odisseo.option_classes.NFWParams>`
     - .. autodoc2-docstring:: odisseo.option_classes.NFWParams
          :summary:
   * - :py:obj:`PointMassParams <odisseo.option_classes.PointMassParams>`
     - .. autodoc2-docstring:: odisseo.option_classes.PointMassParams
          :summary:
   * - :py:obj:`MNParams <odisseo.option_classes.MNParams>`
     - .. autodoc2-docstring:: odisseo.option_classes.MNParams
          :summary:
   * - :py:obj:`SimulationParams <odisseo.option_classes.SimulationParams>`
     - .. autodoc2-docstring:: odisseo.option_classes.SimulationParams
          :summary:
   * - :py:obj:`SimulationConfig <odisseo.option_classes.SimulationConfig>`
     - .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig
          :summary:

API
~~~

.. py:class:: PlummerParams
   :canonical: odisseo.option_classes.PlummerParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.option_classes.PlummerParams

   .. py:attribute:: a
      :canonical: odisseo.option_classes.PlummerParams.a
      :type: float
      :value: 7

      .. autodoc2-docstring:: odisseo.option_classes.PlummerParams.a

   .. py:attribute:: Mtot
      :canonical: odisseo.option_classes.PlummerParams.Mtot
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: odisseo.option_classes.PlummerParams.Mtot

.. py:class:: NFWParams
   :canonical: odisseo.option_classes.NFWParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.option_classes.NFWParams

   .. py:attribute:: Mvir
      :canonical: odisseo.option_classes.NFWParams.Mvir
      :type: float
      :value: None

      .. autodoc2-docstring:: odisseo.option_classes.NFWParams.Mvir

   .. py:attribute:: r_s
      :canonical: odisseo.option_classes.NFWParams.r_s
      :type: float
      :value: 15.3

      .. autodoc2-docstring:: odisseo.option_classes.NFWParams.r_s

   .. py:attribute:: c
      :canonical: odisseo.option_classes.NFWParams.c
      :type: float
      :value: 10

      .. autodoc2-docstring:: odisseo.option_classes.NFWParams.c

   .. py:attribute:: d_c
      :canonical: odisseo.option_classes.NFWParams.d_c
      :type: float
      :value: None

      .. autodoc2-docstring:: odisseo.option_classes.NFWParams.d_c

.. py:class:: PointMassParams
   :canonical: odisseo.option_classes.PointMassParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.option_classes.PointMassParams

   .. py:attribute:: M
      :canonical: odisseo.option_classes.PointMassParams.M
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: odisseo.option_classes.PointMassParams.M

.. py:class:: MNParams
   :canonical: odisseo.option_classes.MNParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.option_classes.MNParams

   .. py:attribute:: M
      :canonical: odisseo.option_classes.MNParams.M
      :type: float
      :value: 65000000000.0

      .. autodoc2-docstring:: odisseo.option_classes.MNParams.M

   .. py:attribute:: a
      :canonical: odisseo.option_classes.MNParams.a
      :type: float
      :value: 3.0

      .. autodoc2-docstring:: odisseo.option_classes.MNParams.a

   .. py:attribute:: b
      :canonical: odisseo.option_classes.MNParams.b
      :type: float
      :value: 0.28

      .. autodoc2-docstring:: odisseo.option_classes.MNParams.b

.. py:class:: SimulationParams
   :canonical: odisseo.option_classes.SimulationParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.option_classes.SimulationParams

   .. py:attribute:: G
      :canonical: odisseo.option_classes.SimulationParams.G
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: odisseo.option_classes.SimulationParams.G

   .. py:attribute:: t_end
      :canonical: odisseo.option_classes.SimulationParams.t_end
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: odisseo.option_classes.SimulationParams.t_end

   .. py:attribute:: Plummer_params
      :canonical: odisseo.option_classes.SimulationParams.Plummer_params
      :type: odisseo.option_classes.PlummerParams
      :value: 'PlummerParams(...)'

      .. autodoc2-docstring:: odisseo.option_classes.SimulationParams.Plummer_params

   .. py:attribute:: NFW_params
      :canonical: odisseo.option_classes.SimulationParams.NFW_params
      :type: odisseo.option_classes.NFWParams
      :value: 'NFWParams(...)'

      .. autodoc2-docstring:: odisseo.option_classes.SimulationParams.NFW_params

   .. py:attribute:: PointMass_params
      :canonical: odisseo.option_classes.SimulationParams.PointMass_params
      :type: odisseo.option_classes.PointMassParams
      :value: 'PointMassParams(...)'

      .. autodoc2-docstring:: odisseo.option_classes.SimulationParams.PointMass_params

   .. py:attribute:: MN_params
      :canonical: odisseo.option_classes.SimulationParams.MN_params
      :type: odisseo.option_classes.MNParams
      :value: 'MNParams(...)'

      .. autodoc2-docstring:: odisseo.option_classes.SimulationParams.MN_params

.. py:class:: SimulationConfig
   :canonical: odisseo.option_classes.SimulationConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig

   .. py:attribute:: N_particles
      :canonical: odisseo.option_classes.SimulationConfig.N_particles
      :type: int
      :value: 1000

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.N_particles

   .. py:attribute:: dimensions
      :canonical: odisseo.option_classes.SimulationConfig.dimensions
      :type: int
      :value: 3

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.dimensions

   .. py:attribute:: return_snapshots
      :canonical: odisseo.option_classes.SimulationConfig.return_snapshots
      :type: bool
      :value: False

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.return_snapshots

   .. py:attribute:: num_snapshots
      :canonical: odisseo.option_classes.SimulationConfig.num_snapshots
      :type: int
      :value: 10

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.num_snapshots

   .. py:attribute:: fixed_timestep
      :canonical: odisseo.option_classes.SimulationConfig.fixed_timestep
      :type: bool
      :value: True

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.fixed_timestep

   .. py:attribute:: num_timesteps
      :canonical: odisseo.option_classes.SimulationConfig.num_timesteps
      :type: int
      :value: 1000

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.num_timesteps

   .. py:attribute:: softening
      :canonical: odisseo.option_classes.SimulationConfig.softening
      :type: float
      :value: 1e-10

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.softening

   .. py:attribute:: integrator
      :canonical: odisseo.option_classes.SimulationConfig.integrator
      :type: int
      :value: None

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.integrator

   .. py:attribute:: acceleration_scheme
      :canonical: odisseo.option_classes.SimulationConfig.acceleration_scheme
      :type: int
      :value: None

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.acceleration_scheme

   .. py:attribute:: batch_size
      :canonical: odisseo.option_classes.SimulationConfig.batch_size
      :type: int
      :value: 10000

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.batch_size

   .. py:attribute:: double_map
      :canonical: odisseo.option_classes.SimulationConfig.double_map
      :type: bool
      :value: False

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.double_map

   .. py:attribute:: external_accelerations
      :canonical: odisseo.option_classes.SimulationConfig.external_accelerations
      :type: tuple
      :value: ()

      .. autodoc2-docstring:: odisseo.option_classes.SimulationConfig.external_accelerations
