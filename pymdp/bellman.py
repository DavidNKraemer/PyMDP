# -*- coding: utf-8 -*-
r"""
    pymdp.bellman
    ~~~~~~~~~~~~~

    Bellman operator and related functions for performing value iteration and
    policy iteration. 

    Consider a Markov Decision Process with state space :math:`X`, action sets
    :math:`A(x)`, transition probabilities :math:`p(y | x, a)`, reward function
    :math:`r(x,a)`, and discount factor :math:`\beta`. For a bounded function
    :math:`f : X \to \mathbb{R}`, the *Bellman operator* of :math:`f` is defined
    as

    .. math:: T_\beta f(x) = \max_{a \in A(x)}\sum_{y \in X} [r(y,a) + \beta f(y)] \cdot p(y|x,a).

    The Bellman operator is an essential component in value and policy
    iteration, both as a theoretical instrument and as a practical one.
"""


from pymdp.mdp import default_value, default_policy


def bellman_step(mdp, value, state, action):
    r"""
    :param mdp: A given Markov decision process object
    :type mdp: MarkovDecisionProcess
    :param value: A value function on the state space of `mdp`
    :type value: dict
    :param state: A state in the state space of `mdp`
    :param action: An action in the action space of `state` in `mdp`

    :return: The numerical result :math:`T^a_\beta v(x)`.
    :rtype: float

    Computes, for a state :math:`x`, action :math:`a`, and given value function
    :math:`v`, the operator

    .. math:: \sum_{y \in X} [r(y,a) + \beta f(y)] \cdot p(y|x,a).

    If we denote this operator by :math:`T_\beta^a`, then it relates to the
    Bellman operator by

    .. math:: T_\beta v(x) = \max_{a \in A(x)} T^a_\beta v(x)
    """
    return sum(
        (mdp.reward(y, action) + mdp.discount * value[y]) * mdp.transition(y, state, action) \
                    for y in mdp.states
        )


def bellman_operator(mdp, value):
    r"""
    :param mdp: A given Markov decision process object
    :type mdp: MarkovDecisionProcess
    :param value: A value function on the state space of `mdp`
    :type value: dict

    :return: The value function and corresponding policy for the Bellman operator :math:`T_\beta v(x)`.
    :rtype: (dict,dict)

    Computes the Bellman operator 

    .. math:: T_\beta v(x) = \max_{a \in A(x)} T^a_\beta v(x)

    for a specified function :math:`v`, and returns a policy which achieves the
    value function :math:`T_\beta v`.
    """
    update = default_value(mdp)
    policy = default_policy(mdp)

    for state in mdp.states:
        policy[state], update[state] = max(
            [(a, bellman_step(mdp, value, state, a)) for a in mdp.actions[state]],
            key=lambda pair: pair[1])

    return policy, update


def bellman_difference(mdp, value, state, action):
    r"""
    :param mdp: A given Markov decision process object
    :type mdp: MarkovDecisionProcess
    :param value: A value function on the state space of `mdp`
    :type value: dict
    :param state: A state in the state space of `mdp`
    :param action: An action in the action space of `state` in `mdp`

    :return: The numerical result :math:`T^a_\beta v(x) - v(x)`.
    :rtype: float

    Computes the function

    .. math:: s_{x,a}(\pi) = T^a v^{\pi}(x) - v^{\pi}(x)

    where :math:`v^{\pi}` is the value function associated with the policy
    :math:`\pi`. The function :math:`s_{x,a}(\pi)` is needed specifically for
    policy iteration.
    """
    return bellman_step(mdp, value, state, action) - value[state]


def generate_policy_from(mdp, value):
    r"""
    Creates a policy corresponding to the given value function.
    """
    return {state:
            next(
                filter(
                    lambda a: bellman_step(mdp, value, state, a) == value[state],
                    mdp.actions[state]
                    )
                ) for state in mdp.states}
