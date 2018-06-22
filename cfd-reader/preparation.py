#! /bin/env/ python3

from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import os
import cv2
import typing
import json

global _home = os.path.expanduser('~')
global _inst = os.path.join(_home, ".cfd-faces")

def is_prepared() -> bool:
    """Checks to see if the installation directory exists and is well"""
    bool = os.path.isdir(_inst)
    for file in {'cfd_stats.csv','images.json'}:
        bool &= os.path.isfile(os.path.join(_inst, file))
    if not bool: print ("Installation is pending, broken, or incomplete.")
    return bool

def prepare() -> :
    os.mkdir(_inst)
    cfddir = input("Please input the full path to the CFD directory that "
        + "directly contains the images directory and the data spreadsheet.\n")
    bool = os.path.isdir(cfddir)
    cfddirls = [os.path.join(cfddir, x) for x in os.listdir(cfddir)]
    imgsdir = list(filter(lambda x: os.path.isdir(x), cfddirls))
    excelsh = list(filter(lambda x: x.endswith('.xlsx'), cfddirls))
    bool &= len(imgsdir) == 1 & len(excelsh) == 1
    assert bool, "Invalid path supplied. Please re-run and supply correct path"\
               + "\nThe path must directly contain the image directory, and "\
               + "the data spreadsheet. If you haven't, please download the "\
               + "Chicago Face Database and extract it to a convenient location"
