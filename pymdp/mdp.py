# -*- coding: utf-8 -*-
r"""
    pymdp.mdp
    ~~~~~~~~~

    Python representation of a Markov decision process, along with functions
    acting on the state space of an MDP.

    A *Markov decision process* (MDP) is formally a tuple

    .. math:: (X, A, r, p, \beta)

    where

    * :math:`X` is the *state space* of the MDP, usually a point set,
    * :math:`A(x)` is, for each state :math:`x \in X`, the *action set* of available actions at :math:`x`,
    * :math:`r(x,a)` is, for each state :math:`x \in X` and :math:`a \in A(x)`, the numerical *one-step reward function* at :math:`x` with action :math:`a`,
    * :math:`p(y|x,a)` is, for states :math:`x,y \in X` and :math:`a \in A(x)`, the *transition probability* of the process arriving at the state :math:`y` from the original state :math:`x` by the action :math:`a`,
    * :math:`\beta` is a number between :math:`0` and :math:`1` indicating the *discount factor*.

"""
from typing import Callable, Iterable, NamedTuple


class MDPFunction(dict):
    """
    Standardized user-facing object for representing functions on the state
    space of an MDP.

    Essentially, an `MDPFunction` is a `dict` whose items can be accessed via
    function calls. This is because many "functions" implemented throughout this
    library are actually `dict` objects *masquerading* as functions. Since they
    should be thought of as functions, this class provides the requisite
    interface to do so.

    We promise that any MDPFunction can access its values via `__call__`, but
    the internal implementation may evolve over time.
    """
    def __call__(self, key):
        r"""
        :param key: The value on which the function is to be evaluated.

        :return: The corresponding result of self[key]
        """
        return self.__getitem__(key)

    def __eq__(self, other):
        return all(self[x] == other[x] for x in self)


MarkovDecisionProcess = NamedTuple(
    "MarkovDecisionProcess",
    [
        ("states", Iterable),
        ("actions", MDPFunction),
        ("reward", Callable),
        ("transition", Callable),
        ("discount", float)
        ]
    )


def default_value(mdp, default=0.):
    r"""
    :param mdp: A given Markov decision process object
    :type mdp: MarkovDecisionProcess

    :param default: The default output of the value function at each point in the state space.
    :type default: float

    :return: The constant value function :math:`v : X \to \mathbb{R}` with the given `v(x) = c`, where :math:`c` was supplied by `default`.
    :rtype: MDPFunction

    Returns the "default" value function, which assigns to each state the value `default` (if unset, this defaults to 0.).
    """
    return MDPFunction({x: default for x in mdp.states})


def default_policy(mdp, default=0):
    r"""
    :param mdp: A given Markov decision process object
    :type mdp: MarkovDecisionProcess
    :param default: The index of the default action for each state in the MDP (defaults to 0)
    :type default: int

    :return: The policy defined by :math:`\pi(x) = A(x)_i` where `i` is the default index supplied.
    :rtype: MDPFunction

    Returns an arbitrary policy, which assigns to each state some available
    action.

    Note that the default policy requires there to be an available action at
    every state. (TODO: generalize this)
    """
    return MDPFunction({x: mdp.actions[x][default] for x in mdp.states})
