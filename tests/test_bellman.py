# -*- coding: utf-8 -*-
"""
Tests of the functions defined in pymdp.bellman
"""


from math import isclose
from pymdp import bellman
from pymdp.mdp import MarkovDecisionProcess, MDPFunction, default_value, default_policy


import pytest
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

transitions = {
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
    """
    Wrapper for the transition function dictionary
    """
    return transitions[(next_state, state, action)]


simple_discount = 1.

@pytest.fixture
def nil_mdp():
    """
    Pytest fixture for an MDP of zero rewards and deterministic actions
    """
    return MarkovDecisionProcess(simple_states, simple_actions, nil_reward,
                                 simple_transition, simple_discount)

@pytest.fixture
def unit_mdp():
    """
    Pytest fixture for an MDP of unit rewards and deterministic actions
    """
    return MarkovDecisionProcess(simple_states, simple_actions, unit_reward,
                                 simple_transition, simple_discount)

@pytest.fixture
def complex_mdp():
    """
    Pytest fixture for an MDP of variable rewards and stochastic actions
    """
    return MarkovDecisionProcess(simple_states, complex_actions, complex_reward,
                                 complex_transition, simple_discount)


@pytest.fixture
def zeros_value():
    """
    Value function of all zeros
    """
    return MDPFunction({x: 0. for x in simple_states})


@pytest.fixture
def ones_value():
    """
    Value function of all ones
    """
    return MDPFunction({x: 1. for x in simple_states})


@pytest.fixture
def zeros_policy():
    """
    Policy function of all zero-states
    """
    return MDPFunction({x: 0 for x in simple_states})


def test_bellman_step(nil_mdp, unit_mdp, complex_mdp, zeros_value, ones_value):
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

def test_bellman_operator(nil_mdp, unit_mdp, complex_mdp, zeros_value, ones_value, zeros_policy):
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


def test_bellman_difference(nil_mdp, unit_mdp, complex_mdp):
    pass
