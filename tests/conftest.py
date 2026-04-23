"""Shared fixtures for the eml test suite."""
import math
import pytest
from eml_math import EMLPoint, EMLState


@pytest.fixture
def unit_point():
    return EMLPoint(1.0, 1.0)


@pytest.fixture
def unit_knot(unit_point):
    return EMLState(unit_point)


@pytest.fixture
def d100_point():
    return EMLPoint(1.0, 1.0, D=100)


@pytest.fixture
def d100_knot(d100_point):
    return EMLState(d100_point)
