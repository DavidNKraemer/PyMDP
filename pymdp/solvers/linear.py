# -*- coding: utf-8 -*-
r"""
    pymdp._linear_programming
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Implementation of linear programming solution to the infinite horizon
    problem.
"""


from pymdp.mdp import MarkovDecisionProcess
from pymdp.bellman import generate_policy_from
from scipy.optimize import linprog
from numpy import array


def linear_programming(mdp: MarkovDecisionProcess):
    """
    Linear programming
    """

    def constraint_coefficient(next_state, state, action):
        """
        Helper function
        """
        offset = 1. if next_state == state else 0.
        return offset - mdp.discount * mdp.transition(next_state, state, action)

    objective_vector = [1. for x in mdp.states]

    constraint_matrix = [[-constraint_coefficient(y, x, a) for y in mdp.states] \
            for x in mdp.states for a in mdp.actions[x]]

    constraint_vector = [-mdp.reward(x, a) for x in mdp.states for a in mdp.actions[x]]

    print(array(constraint_matrix))
    print(array(constraint_vector))

    print("made it this far!")
    results = linprog(objective_vector, A_ub=constraint_matrix, b_ub=constraint_vector)
    value = {state: results.x[i] for (i, state) in enumerate(mdp.states)} 
    # policy = generate_policy_from(mdp, value)
    return value #, policy
