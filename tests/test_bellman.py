# -*- coding: utf-8 -*-
"""
Tests of the functions defined in pymdp.bellman
"""


from math import isclose
from pymdp import bellman
from pymdp.mdp import MarkovDecisionProcess, MDPFunction, default_value, default_policy


import codecov

simple_states = [0, 1]
simple_actions = {0: [0], 1: [0]}
complex_actions = {x: [0, 1] for x in simple_states}

def nil_reward(state, action):
    return 0.

def unit_reward(state, action):
    return 1.

def complex_reward(state, action):
    return float(state) + float(action)

def simple_transition(next_state, state, action):
    return 1. if (next_state == 1 - state) else 0.

comtrans = {
    (1, 0, 0): 0.3,
    (0, 0, 0): 0.7,
    (1, 1, 0): 0.6,
    (0, 1, 0): 0.4,
    (1, 0, 1): 0.25,
    (0, 0, 1): 0.75,
    (1, 1, 1): 0.5,
    (0, 1, 1): 0.5
    }
def complex_transition(next_state, state, action):
    return comtrans[(next_state, state, action)]


simple_discount = 1.
nil_mdp = MarkovDecisionProcess(simple_states, simple_actions, nil_reward,
        simple_transition, simple_discount)
unit_mdp = MarkovDecisionProcess(simple_states, simple_actions, unit_reward,
        simple_transition, simple_discount)
complex_mdp = MarkovDecisionProcess(simple_states, complex_actions,
        complex_reward, complex_transition, simple_discount)

zeros_value = default_value(nil_mdp, default=0.)
ones_value = default_value(nil_mdp, default=1.)

zeros_policy = default_policy(nil_mdp, default=0)



def test_bellman_step():
    """
    Do a thing
    """
    assert bellman.bellman_step(nil_mdp, zeros_value, 0, 0) == 0.
    assert bellman.bellman_step(nil_mdp, ones_value, 0, 0) == 1.

    assert bellman.bellman_step(unit_mdp, zeros_value, 0, 0) == 1.
    assert bellman.bellman_step(unit_mdp, ones_value, 0, 0) == 2.

    assert isclose(bellman.bellman_step(complex_mdp, ones_value, 0, 0), 1.3)
    assert isclose(bellman.bellman_step(complex_mdp, ones_value, 0, 1), 2.25)
    assert isclose(bellman.bellman_step(complex_mdp, ones_value, 1, 0), 1.6)
    assert isclose(bellman.bellman_step(complex_mdp, ones_value, 1, 1), 2.5)

def test_bellman_operator():
    """
    Some tests
    """
    policy, value = bellman.bellman_operator(nil_mdp, zeros_value)
    assert value == zeros_value
    assert policy == zeros_policy

    policy, value = bellman.bellman_operator(nil_mdp, ones_value)
    assert value == ones_value
    assert policy == zeros_policy

    policy, value = bellman.bellman_operator(unit_mdp, ones_value)
    assert value == default_value(unit_mdp, default=2.)
    assert policy == zeros_policy

    policy, value = bellman.bellman_operator(complex_mdp, ones_value)
    assert value == MDPFunction({0: 2.25, 1: 2.5})
    assert policy == default_policy(complex_mdp, default=1)


def test_bellman_difference():
    pass

    

