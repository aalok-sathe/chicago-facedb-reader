from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import os

from . import preparation
from . import faces

global home = os.path.expanduser('~')
global inst = os.path.join(home, ".cfd-faces")

try:
    assert preparation.is_prepared(home=home, inst=inst)
except AssertionError:
    preparation.prepare(home=home, inst=inst)
