# -*- coding: utf-8 -*-

import pytest
from mercs.skeleton import fib

__author__ = "Elia vw"
__copyright__ = "Elia vw"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
