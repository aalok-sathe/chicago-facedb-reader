#! /bin/env/ python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

global _home = os.path.expanduser('~')
global _inst = os.path.join(_home, ".cfd-faces")

def is_prepared():
    bool = os.path.isdir(_inst)
    if bool:
        for file in {'cfd_stats.csv','images.pickle'}
        bool = os.path.isfile(os.path.join(_inst, file))
