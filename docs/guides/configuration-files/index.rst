.. currentmodule:: academia.curriculum

.. _config-files:

Configuring tasks & curricula
-----------------------------

Intro
=====

In *academia* package, there are two ways of initializing tasks and curricula.
The first method is through the use of :class:`LearningTask` and
:class:`Curriculum`'s constructors. The other utilizes configuration files in the YAML
format and :func:`load_task_config` or :func:`load_curriculum_config` functions.

Here is an example of initializing a simple curriculum directly in a script:

.. literalinclude:: codes/1-code-init.py
  :linenos:

This code creates a curriculum which comprises the first two levels of the Lava Crossing
environment. An identical curriculum can be defined with the following configuration file:

.. literalinclude:: codes/2/my_config.curriculum.yml
  :language: YAML
  :linenos:
  :caption: my_config.curriculum.yml

This can be then loaded with a single line of code:

.. literalinclude:: codes/2/load.py
  :linenos:

Neither method is better than the other and it is up to the user to choose which
one they prefer.
Initializing through code gives more flexibility and can be easier for users not
familiar with *academia*'s API. On the other hand, configuration files allow to
extract most of the configuration logic out of the source code. They can also make large
and complex configurations more concise and readable, which might make them
a better option for more complex experiments.

To learn about the specific parameters for environments, tasks and curricula,
feel free to explore the rest of the documentation to get familiar with
*academia*'s functions and classes. The rest of this guide will focus on YAML
configuration files. More specifically, we will explore some special features
which make this method flexible and allow users to avoid duplication in
their configuraions.

.. note::
   While the configuration has to be in the YAML format, *academia* does not
   enforce any particular file extensions. However, it is often a good practise
   to differentiate task and curricula configuration files by using extensions
   such as ``.task.yml`` or ``.curriculum.yml``.

Default task parameters inside a curriculum
===========================================

Tasks inside a single curriculum often share similar sets of parameter values.
For example, all of them could utilize the same environment, but with
different difficulty levels. Curriculum configuration file allows to
define a set of default parameters for tasks inside that curriculum.

In the example configuration above, both tasks share a lot of the same
configuration, which leads to lots of code duplication.
Below are highlighted the only unique pieces of configuration for
both tasks:

.. literalinclude:: codes/2/my_config.curriculum.yml
  :language: YAML
  :linenos:
  :emphasize-lines: 8,14,17,23

To address this issue, a special ``_default`` task can be defined for
the curriculum, which provides default parameters values for all tasks
defined or loaded in this curriculum (more on loading later). The
configuration listed above can be simplified in the following way:

.. literalinclude:: codes/3-default-excluded.yml
  :language: YAML
  :linenos:

Now, all common configuration has been moved to the ``_default`` task, and
the tasks define only their unique arguments. Note that the ``_default`` task
can also be used in the curriculum, just as any other task. All we need to do
is to supply all required parameters to it. Consider the following configuration,
which again is equivalent to the ones listed before:

.. literalinclude:: codes/4-default-included.yml
  :language: YAML
  :linenos:

In curriculum learning, the final environment is treated as the most important one,
and all other tasks are only there to speed up the training. It makes sense then to
mark the target environment as ``_default`` in the configuration, and then for easier
tasks define just their unique pieces of configuration. This is exactly what we do
in the above example. Notice that both ``_default`` and ``easier`` tasks define
the environment difficulty, as well as a max episodes stop condition. Each task can
override the default configuration, and this is exactly what happens here.
For instance, the ``easier`` task is now going to end after 500 episodes -
if we did not specify this stop condition here, it would end after 1000 episodes,
just as declared in the ``_default`` task.

Loading configurations from external files
==========================================

It is not uncommon for multiple curricula to share common tasks. Let us say we want to
design two curricula for the Door Key environment. Consider the difficulty level
of 2 as the target difficulty for this environment. In the first curriculum, we want
an agent to go through all the difficulty levels up to the level 2, starting at level 0.
In the other curriculum, we want it to skip the level 1 and go straight from level 0 to
level 2. Below are example configurations for both scenarios:

.. literalinclude:: codes/5/full.curriculum.yml
  :language: YAML
  :linenos:
  :caption: full.curriculum.yml

.. literalinclude:: codes/5/task-skip.curriculum.yml
  :language: YAML
  :linenos:
  :caption: task-skip.curriculum.yml

We use the ``_default`` task to avoid configuration duplication in each of the
files. Still, the configurations for tasks named "Easy task" and "Hard task" are identical
in both files. It would be nice to somehow extract it to a separate file, and load it in
both of the above's configurations. Luckily, we can do it using the special attribute
named ``_load``. It tells the configuration loaders to load YAML attributes from another
file. This way, we can split the above configurations into multiple files to create
an equivalent configuration:

.. literalinclude:: codes/6/easy.task.yml
  :language: YAML
  :linenos:
  :caption: easy.task.yml

.. literalinclude:: codes/6/intermediate.task.yml
  :language: YAML
  :linenos:
  :caption: intermediate.task.yml

.. literalinclude:: codes/6/hard.task.yml
  :language: YAML
  :linenos:
  :caption: hard.task.yml

.. literalinclude:: codes/6/full.curriculum.yml
  :language: YAML
  :linenos:
  :caption: full.curriculum.yml
  :emphasize-lines: 14,16,18

.. literalinclude:: codes/6/task-skip.curriculum.yml
  :language: YAML
  :linenos:
  :caption: task-skip.curriculum.yml
  :emphasize-lines: 13,15

Note that the path provided for the ``_load`` attribute must be relative to the
current configuration file.

The ``_load`` special attribute can be used not just to load tasks. It is designed to be
able to load attributes from any YAML file, which makes it very versatile. For example,
in the above configurations, since the ``_default`` task is also shared across both curricula,
we could extract its parameters into a separate file. It could look as follows
for full curriculum (analogously for the task-skip curriculum):

.. literalinclude:: codes/7/task-defaults.yml
  :language: YAML
  :linenos:
  :caption: task-defaults.yml

.. literalinclude:: codes/7/full.curriculum.yml
  :language: YAML
  :linenos:
  :caption: full.curriculum.yml
  :emphasize-lines: 7

The ``_load`` special attribute could also be chained, i.e. you can load a file, which
has ``_load`` in it, and it will also be handled. Also, just like with the ``_default``
task, attributes loaded with the ``_load`` can be overriden if you specify them
alongside the ``_load`` attribute:

.. literalinclude:: codes/8-full-override.curriculum.yml
  :language: YAML
  :linenos:
  :caption: full.curriculum.yml
  :emphasize-lines: 8,9

This is just one way to transform these configurations, and there could possibly be even
better ways to structure them. Remember that the ``_load`` special attribute can be used in
both tasks and curricula configurations.

Variables in configuration files
================================

So far all configuration files we looked at had all the parameter values hardcoded. There
could be cases however when we might want to input some of the parameters dynamically.
For example, let us say we want to run a task 10 times to be able to average the results
of our experiment across different independent runs. Consider the following configuration
file and script:

.. literalinclude:: codes/9/doorkey.task.yml
  :language: YAML
  :linenos:
  :caption: doorkey.task.yml

.. literalinclude:: codes/9/run.py
  :linenos:
  :caption: run.py

Note that we specify a random state to the Door Key environment to ensure reproducibility
of our experiments. However, it could be better to pass a different random seed to the
environment for each individual run. We can achieve this using variables inside our
configuration. Variables are marked by a dollar sign ``$`` in the configuration files
and can be used as follows:

.. literalinclude:: codes/10/doorkey.task.yml
  :language: YAML
  :linenos:
  :caption: doorkey.task.yml
  :emphasize-lines: 5

.. literalinclude:: codes/10/run.py
  :linenos:
  :caption: run.py
  :emphasize-lines: 7,8,9

The same syntax applies for the :func:`load_curriculum_config` function. Variables can also be
used in external files loaded via the ``_load`` attribute - the same ``variables`` dictionary
will be used to resolve variables in any loaded files.

Variables can also be useful in setting parameters which are not possible to be set directly
in the configuration files. Good examples of such parameters are ``task_callback`` for
:class:`Curriculum` and ``episode_callback`` for :class:`LearningTask`. In the following
example, we use a variable to configure the former:

.. literalinclude:: codes/11/my_curriculum.yml
  :language: YAML
  :linenos:
  :caption: my_curriculum.yml
  :emphasize-lines: 2

.. literalinclude:: codes/11/run.py
  :linenos:
  :caption: run.py
  :emphasize-lines: 9,10,11

These examples provide just the most common use cases. Variables have
also been designed with versitality in mind, and could also be used to
specify full tasks inside a curriculum, or to order tasks in a curriculum.
Basically, any attribute in the configuration (except for ``_load``) can have a
variable assigned to it with a value provided at runtime upon loading.

.. note::
   Variables cannot be used to dynamically provide paths for the ``_load`` attribute.
   This is because by design all loads are handled before variables are resolved.
