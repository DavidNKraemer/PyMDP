# -*- coding: utf-8 -*-
"""
    pymdp.solvers
    ~~~~~

    An implementation of standard algorithms for Markov decision processes in
    Python.

    :copyright: (c) 2018 by David Kraemer
    :license: MIT, see LICENSE for more details
"""

from pymdp.solvers.value import value_iteration
from pymdp.solvers.policy import policy_iteration
from pymdp.solvers.linear import linear_programming
