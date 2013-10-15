# init for externals package
""" Calls to external packages """

from dipy.external.fsl import pipe

# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester
