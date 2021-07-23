# pypi
from pytest import fixture

class Katamari:
    """Stick things here"""

@fixture
def katamari():
    return Katamari()
