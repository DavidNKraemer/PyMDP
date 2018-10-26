# -*- coding utf-8 -*-
r"""
    pymdp.utils
    ~~~~~~~~~~~

    Functions which have uses throughout the library. In particular, the sup
    norm :math:`\|f\|_{\infty} = \sup_{x \in X} |f(x)|` for bounded functions
    :math:`f : X \to \mathbb{R}` on finite spaces.
"""


def _compose(outer, inner):
    return lambda x: outer(inner(x))


def sup_norm(space, fun):
    r"""
    :param space: A collection of points.
    :type space: container class
    :param fun: A function defined on space.
    :type fun: `space` -> float

    :return norm: the value of the supremum norm.
    :rtype: float

    Computes the supremum norm of the function :math:`f : X \to \mathbb{R}`
    defined by

    .. math:: \| f \|_{\infty} &= \sup_{x \in X} | f(x) | \\
                               &= \max_{x \in X} | f(x) |

    Since :math:`X` is assumed finite, the supremum can be replaced with a
    maximum.
    """
    return max(map(_compose(abs, fun), space))


def sup_distance(space, fun1, fun2):
    r"""
    :param space: A collection of points.
    :type space: container class
    :param fun1: A function defined on space.
    :type fun1: `space` -> float
    :param fun2: A function defined on space.
    :type fun2: `space` -> float

    :return distance: the value of the supremum distance
    :rtype: float

    Given two functions f,g which operate on the state space of a given MDP,
    computes

    .. math:: \|f-g\| &= \sup_{x \in X} |f(x) - g(x)| \\
                      &= \max_{x \in X} |f(x) - g(x)|
    """
    return sup_norm(space, lambda x: fun1(x) - fun2(x))
