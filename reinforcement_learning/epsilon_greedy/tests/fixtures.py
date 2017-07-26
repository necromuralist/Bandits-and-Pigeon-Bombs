# pypi
import pytest

class Context(object):
    """something to hold values"""


@pytest.fixture
def context():
    return Context()
