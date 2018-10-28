# -*- coding: utf-8 -*-
"""
Tests of the functions defined in pymdp.utils
"""


from pymdp import utils


import codecov


def test_sup_norm():
    """
    Tests for the function utils.sup_norm
    """
    # test a constant function on a mixed space
    fun = lambda x: 1
    space = [1, 2, 'a', fun]
    assert utils.sup_norm(space, fun) == 1

    # test a function which takes on both positive and negative values
    fun = lambda x: x ** 2
    space = range(-10, 4)
    assert utils.sup_norm(space, fun) == 100

    # exotic functions. here's an indicator on strings, over a string
    fun = lambda x: 1 if isinstance(x, str) else 0
    space = "each character is just a singleton strings"
    assert utils.sup_norm(space, fun) == 1


def test_sup_distance():
    """
    Tests for the function utils.sup_distance
    """
    fun = lambda pair: pair[0] - pair[1]
    gun = lambda pair: pair[0] + pair[1]
    space = [(x, y) for x in range(10) for y in range(-9, 1)]
    assert utils.sup_distance(space, fun, gun) == 18

    fun = lambda x: 2
    gun = lambda x: 1
    space = [1, 2, 3, 4, 5, type, '$f(x) = 2, g(x) = 1$']
    assert utils.sup_distance(space, fun, gun) == 1

    fun = len
    gun = lambda string: 0
    space = " ".split("when in the course of human events, it becomes necessary")
    assert utils.sup_distance(space, fun, gun) == utils.sup_norm(space, fun)



def triangle_test(space, fun1, fun2):
    """
    Wrapper for the triangle inequality
    """
    return utils.sup_distance(space, fun1, fun2) <= \
       utils.sup_norm(space, fun1) + utils.sup_norm(space, fun2)


def test_triangle_inequality():
    """
    For any functions f, g, we should have ||f - g| <= ||f|| + ||g||
    """
    fun = lambda x: 1
    gun = lambda x: 2
    space = [1, 2, str, [1, 'a'], {}]
    assert triangle_test(space, fun, gun)
