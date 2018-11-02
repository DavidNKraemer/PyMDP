# -*- coding: utf-8 -*-
r"""
    pymdp.solvers.value
    ~~~~~~~~~~~~~~~~~~~

    Implementation of value iteration for the infinite horizon problem.
"""
from pymdp.bellman import bellman_operator
from pymdp.mdp import MarkovDecisionProcess, default_value
from pymdp.utils import sup_distance


def value_iteration(mdp: MarkovDecisionProcess, epsilon):
    """
    Performs infinite horizon value iteration for a given MDP with a numerical
    error term epsilon.
    """
    value = default_value(mdp)
    tolerance = (1. - mdp.discount) / (2. * mdp.discount) * epsilon
    proceeding = True

    while proceeding: 
        policy, update = bellman_operator(mdp, value)
        if sup_distance(mdp.states, update, value) > tolerance:
            value = update
        else:
            proceeding = False

    return policy, value
