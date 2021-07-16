.. title: PyBrain Optimization Example
.. slug: PyBrain-Optimization-Example
.. date: 2018-01-12 16:16:00
.. tags: pybrain optimization
.. link: 
.. description: Trying out the pybrain Hill Climbing optimization with Evolvable.
.. type: text
.. author: necromuralist

This is the `general optimization example <http://www.pybrain.org/docs/tutorial/optimization.html#general-optimization-using-evolvable>`_ from the pybrain documentation.

1 Imports
---------

.. code:: python

    # python standard library
    from random import random

    # pypi
    from pybrain.structure.evolvables.evolvable import Evolvable
    from pybrain.optimization import HillClimber

2 The Evolvable
---------------

To make an optimization that can take arbitrary values (not just continuous numbers), you can implement a sub-class of the PyBrain `Evolvable <http://www.pybrain.org/docs/api/structure/evolvables.html>`_ class.

2.1 The Constructor
~~~~~~~~~~~~~~~~~~~

The ``Evolvable`` class doesn't implement a constructor so you can create one with any parameters you need.

2.2 The Mutate Method
~~~~~~~~~~~~~~~~~~~~~

This is the method that is called after each round to change the parameters a little (a `tweak`). It takes positional arguments, but I think it's called by the Hill Climber so I don't know where it gets passed in.

2.3 The Copy Method
~~~~~~~~~~~~~~~~~~~

The tutorial says this is a required method, but the documentation for the API says it should default to a deep-copy. Anyway, I think this is only used if you use something like a Genetic Algorithm.

2.4 The Randomize Method
~~~~~~~~~~~~~~~~~~~~~~~~

This is used to initialize the parameters to a random value. This is required but I'm pretty sure it doesn't get used in this case.

.. code:: python

    class Mutant(Evolvable):
        """A simple evolvable class

        Args:
         x: a starting value to mimic the fitness of the model
         mininmum: smallest allowed value
         maximum: biggest allowed value
        """
        def __init__(self, x, minimum=0, maximum=10):
            self.minimum = minimum
            self.maximum = maximum
            # minimum <= x <= maximum
            self._x = None
            self.x = x
            return

        @property
        def x(self):
            """The value to optimize
        
            Returns:
             x (float): value to optimize
            """
            return self._x

        @x.setter
        def x(self, new_x):
            """sets x, constraining the value

            Args:
             new_x: float from minimum to maximum
            """
            self._x = max(self.minimum, min(new_x, self.maximum))
            return

        def mutate(self):
            """Updates x with a random change
        
            Maintains the constraint of the value
            """
            self.x += (random() - 0.3)
            return

        def copy(self):
            """Returns a new instance with the same x-value

            Returns:
             Mutant: copy of this instance
            """
            return Mutant(self.x)

        def randomize(self):
            """A method to randomize the x-value"""
            self.x = self.maximum * random()
            return

        def __repr__(self):
            """String representation
        
            Returns:
             str: formatted version of x
            """
            return "< {:.2f} (Maximized={})>".format(self.x, self.x == self.maximum)

3 Hill Climbing
---------------

The `HillClimber <http://www.pybrain.org/docs/api/optimization/optimization.html#module-pybrain.optimization>`_ is the simplest search - it assumes the first minima or maxima it finds is the global one. By default it tries to maximize the outcome. None of the arguments are required at instantiation, but in this case we're setting:

- an ``evaluator``: a callable that outputs how well the object to be evaluated did

- an ``evaluable``: the object to be evaluated in this case our ``Mutant``

- ``maxEvaluations``: The maximum number of times the ``evaluable`` is evaluated

- ``verbose``: print each step

- ``desiredEvaluation``: the value that is good enough so the climber can stop

3.1 The Evaluator Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case we're just going to return the x value of the object.

.. code:: python

    def evaluator(mutant):
        return mutant.x

3.2 The Instances
~~~~~~~~~~~~~~~~~

.. code:: python

    mutant = Mutant(random() * 10)
    climber = HillClimber(evaluator, mutant, maxEvaluations=50, verbose=True, desiredEvaluation=10)

3.3 The Optimization
~~~~~~~~~~~~~~~~~~~~

The optimization classes get run using their ``learn`` methods.

.. code:: python

    outcome = climber.learn()
    print(outcome)

::

    ('Step:', 0, 'best:', 6.780765339892317)
    ('Step:', 1, 'best:', 6.780765339892317)
    ('Step:', 2, 'best:', 6.807553650921801)
    ('Step:', 3, 'best:', 7.282574697921699)
    ('Step:', 4, 'best:', 7.45592511459156)
    ('Step:', 5, 'best:', 7.533694376079802)
    ('Step:', 6, 'best:', 7.751507552794123)
    ('Step:', 7, 'best:', 8.184303418505593)
    ('Step:', 8, 'best:', 8.184303418505593)
    ('Step:', 9, 'best:', 8.224264996606221)
    ('Step:', 10, 'best:', 8.4835021736195)
    ('Step:', 11, 'best:', 9.153976071682798)
    ('Step:', 12, 'best:', 9.55795557780446)
    ('Step:', 13, 'best:', 10)
    (< 10.00 (Maximized=True)>, 10)

It managed to find the maximum in 13 steps.
