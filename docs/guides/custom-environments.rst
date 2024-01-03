.. currentmodule:: academia.environments.base

.. _custom-envs:

Using your own environments
---------------------------

To use a custom environment with *academia* package, a wrapper needs to be
created to make the environment compatible with package's environment API.
This can be achieved by extending one of the classes provided by this module.

The most basic class for environments is
:class:`ScalableEnvironment`, which defines the environment API used throughout
*academia* package. However there are also some generic wrappers which can
speed up the process of creating wrappers for custom environments.

If your environment is compliant with *Gymnasium*'s API, check out
:class:`GenericGymnasiumWrapper`. It contains implementations for all methods
exposed by *academia*'s environment API. You only need to take care of a few
things:

- Override the constructor and handle the ``difficulty`` setting logic there;

- Provide an implementation for a protected ``_transform_state`` method,
  which transforms a state returned by the underlying environment to a *NumPy*
  array, which is the format used in *academia* package.

- (Optional) Provide an implementation for a protected ``_transform_state``
  method to make sure that there are no unused states (see
  :class:`academia.environments.DoorKey`'s source code for an example use case).

The two other generic wrappers, :class:`GenericMiniGridWrapper` and
:class:`GenericAtariWrapper` work similarly but have some extra functionalities
which can help to set up some more specific environments.
The former makes it easier to handle the ``difficulty`` parameter for
*MiniGrid* environments while the latter provides a default
``_transform_state`` implementation for *Gymnasium*'s Atari environments.

Feel free to browse package's source code to see how these base classes are
used in practise.
