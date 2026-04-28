"""Shared fixtures for the eml-math test suite."""
import math
import pytest
from eml_math import EMLPoint


@pytest.fixture
def unit_point():
    return EMLPoint(1.0, 1.0)


@pytest.fixture
def d100_point():
    return EMLPoint(1.0, 1.0, D=100)
