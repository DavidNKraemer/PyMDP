# -*- coding: utf-8 -*-
"""
    pymdp
    ~~~~~

    An implementation of standard algorithms for Markov decision processes in
    Python.

    :copyright: (c) 2018 by David Kraemer
    :license: MIT, see LICENSE for more details
"""


from pymdp.mdp import MarkovDecisionProcess
from pymdp.solvers import value_iteration, policy_iteration

__version__ = '0.1.0'
