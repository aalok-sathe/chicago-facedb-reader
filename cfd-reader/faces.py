#! /bin/env/ python3

from __future__ import absolute_import
# from __future__ import print_function # python3 print functionality
# from __future__ import division # python3 float division functionality

import os # for filesystem access
import cv2 # opencv library for image processing
import typing # for function annotations
import json # to dump objects in the json format (as opposed to pickled binary)
import xlrd,csv # to read the CFD data codebook and output it to a csv file
from progressbar import progressbar # to display progress during iteration
from collections import defaultdict # to use during indexing faces
import numpy as np # for multidimensional arrays and vector arithmetic

def supply_dict(key = 'genders',
                legend = {'genders' : 0, 'emotions' : 1}):
    """constructs a nested disctionary with appropriate cetegory hierarchy for
       convenient access later on"""
    if legend[key] == 0:
        return dict([
            ('M', supply_dict('emotions')),
            ('F', supply_dict('emotions')),
        ])
    else:
        return dict([
            ('N', defaultdict(str)),
            ('F', defaultdict(str)),
            ('A', defaultdict(str)),
            ('HO', defaultdict(str)),
            ('HC', defaultdict(str)),
        ])

class Race(enum.Enum):
    A = 1; B = 2; L = 3; W = 4
    UNK = 0

class Gender(enum.Enum):
    F = 1; M = 2
    UNK = 0

class Emotion(enum.Enum):
    A = 1; F = 2; HC = 3; HO = 4; N = 5
    UNK = 0

def index_faces(imgsdir=None, inst=None):
    """Processes and indexes faces from the CFD and dumps them to a json file
       for easy retrieval"""
    # Iterate over subfolders corresponding to each person in the DB
    for container_name in progressbar(self.image_containers, redirect_stdout=True):
        # Skip invalid directories
        if len(container_name)<1 or len(container_name.split('-'))<2:
            print("ignoring", container_name)
            continue
        # First two characters of dir name are race, gender. E.g. AF
        rac, gen = container_name[:2]
        # Now iterate over the individual photos of each person
        for filename in os.listdir(os.path.join(os.path.abspath(path), container_name)):
            basename = filename.split('.')[0]
            if not len(basename):
                continue
            id,emo = basename.split('-')[2], basename.split('-')[4]
            print(os.path.join(os.path.abspath(path), filename))
            # Store image in a central dict
            self.images[rac][gen][emo][id] = cv2.imread(os.path.join(os.path.abspath(path), container_name, filename))
            # Crop to a square according to lowest of width or height
            self.images[rac][gen][emo][id] = self.crop_square(rac=rac, gen=gen, emo=emo, id=id)
            # Resize
            self.images[rac][gen][emo][id] = self.resize(rac=rac, gen=gen, emo=emo, id=id, resize=(32,32))
            # Add unique identifier to a set for later iteration
            self.indexed_faces.add(rac+' '+gen+' '+emo+' '+id)
