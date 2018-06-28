#! /bin/env/ python3

from __future__ import absolute_import
# from __future__ import print_function # python3 print functionality
# from __future__ import division # python3 float division functionality

import os # for filesystem access
import cv2 # opencv library for image processing
import typing # for function annotations
import json # to dump objects in the json format (as opposed to pickled binary)
import pickle # to dump objects as an alternative to json
import xlrd,csv # to read the CFD data codebook and output it to a csv file
from progressbar import progressbar # to display progress during iteration
from collections import defaultdict # to use during indexing faces
import numpy as np # for multidimensional arrays and vector arithmetic
import enum # for enumerating labels using indices

################################################################
# a helper method returning a dict object for use in indexing
################################################################
def supply_dict(key = 'genders',
                legend = {'genders' : 0, 'emotions' : 1}):
    """constructs a nested disctionary with appropriate cetegory hierarchy for
       convenient access later on"""
    if legend[key] == 0:
        return dict([
            ('M', supply_dict('emotions')),
            ('F', supply_dict('emotions')),
        ])
    return dict([
        ('N', defaultdict(str)),
        ('F', defaultdict(str)),
        ('A', defaultdict(str)),
        ('HO', defaultdict(str)),
        ('HC', defaultdict(str)),
    ])

################################################################
# some enums to number labels while returning label vectors
################################################################
class Race(enum.Enum):
    A = 1; B = 2; L = 3; W = 4
    UNK = 0
class Gender(enum.Enum):
    F = 1; M = 2
    UNK = 0
class Emotion(enum.Enum):
    A = 1; F = 2; HC = 3; HO = 4; N = 5
    UNK = 0

################################################################
# crops a given image to square dimensions by reducing Whichever
# one of height or width is greater, equally on both sides
################################################################
def crop_img(img=None) -> np.ndarray:
    """Crop image to a square with dimension that is lowest
    of height and width. Whichever one of those dimensions is
    greater is reduced to the newly determined dimension of
    the square, using half-delta reduction from two ends"""
    try:
        h,w = img.shape[0:2]
        d = 0.5 * abs(h-w)
        return img[int((h>w)*d):int(h-(h>w)*d), int((w>h)*d):int(w-(w>h)*d)]
    except AttributeError:
        print("Error, passed argument is not a proper image.")
        return img

################################################################
# resizes image to supplied 2D shape (doesn't disturb channels)
################################################################
def resize_img(img=None, resize=(32,32)) -> np.ndarray:
    """Resize image to supplied dimensions"""
    if resize != img.shape[0:2]:
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
    return img

################################################################
# make a `Face` class as an option for returning, with its own
# methods for data access, so that data can be returned in a
# more organized manner
################################################################
class Face:
    path = None
    resize = None
    shape = None
    imgdata = None
    rac = None; gen = None; emo = None; id = None

    def __init__(self, imgdata=None, path=None, resize=None, shape=None,
                 grayscale=1, cache=False, rac=None, gen=None, emo=None,
                 id=None):
        self.imgdata = imgdata
        self.path = path
        self.resize = resize
        self.shape = shape
        self.rac = rac; self.gen = gen; self.emo = emo; self.id = id

        if cache:
            self.imgdata = cv2.imread(self.path)
            self.shape = self.imgdata.shape

    def get_channels(self):
        return self.shape[2]

    def set_img(self, imgdata=None):
        self.imgdata=imgdata
        self.shape=imgdata.shape

    def retrieve_img(self, resize=resize, crop_square=True, grayscale=True):
        if type(self.imgdata) != np.array:
            self.imgdata = cv2.imread(self.path)
        img = self.imgdata
        if resize != None:
            img = resize_img(img, resize=resize)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (*img.shape[0:2], 1))
        if crop_square:
            img = crop_img(img)
        return img

    def save_img(self, new_path=None):
        cv2.imwrite(new_path, self.imgdata)

################################################################
# index and process faces according to supplied options
################################################################
def index_faces(imgsdir=None, inst=None, img_containers=None, cache=True, crop_square=True, resize=None, verbose=True) -> dict:
    """Processes and indexes faces from the CFD and dumps them to a json file
       for easy retrieval"""
    # a dictionary to index references to images
    img_ref_dict = {
        'A' : supply_dict(), 'W' : supply_dict(),
        'B' : supply_dict(), 'L' : supply_dict(),
    }
    indexed_faces = set()
    # iterate over subfolders corresponding to each person in the DB
    for container in progressbar(img_containers, redirect_stdout=True):
        # iterate over the individual pictures of each person
        if verbose: print("reading from %s" % container)
        for filename in os.listdir(container):
            basename = filename.split('.')[0]
            if not len(basename):
                continue
            # note down person ID and the facial expression (emotion)
            id,emo = basename.split('-')[2], basename.split('-')[4]
            # assign race, gender. E.g. AF
            rac,gen = basename.split('-')[1][:]
            # add unique identifier to a set for later iteration
            indexed_faces.add(rac+' '+gen+' '+emo+' '+id)
            # store image reference in a central dict
            face = Face(rac=rac, gen=gen, emo=emo, id=id, resize=resize,
                        path=os.path.join(container, filename), cache=True)

            if verbose: print("Processing image:",rac+' '+gen+' '+emo+' '+id)

            img_ref_dict[rac][gen][emo][id] = face
            # os.path.join(container, filename)

            if crop_square:
                # crop to a square according to lowest of width or height
                face.set_img(crop_img(img=face.imgdata))
            if resize != None:
                # Resize
                face.set_img(resize_img(img=face.imgdata, resize=resize))
            if cache:
                instimgdir = os.path.join(inst, 'images')
                face.save_img(os.path.join(instimgdir,
                                           rac+' '+gen+' '+emo+' '+id+'.png'))

    # dump objects to .pickle files in the installation directory
    with open(os.path.join(inst,'images.pickle'),'wb') as imgout:
        pickle.dump(img_ref_dict, imgout)
        if verbose: print("Pickle output at %s"%str(imgout))
    with open(os.path.join(inst,'indexed.pickle'),'wb') as indout:
        pickle.dump(indexed_faces, indout)
        if verbose: print("Pickle output at %s"%str(indout))

    return img_ref_dict, indexed_faces

################################################################
# getter method to retrieve the image of a particular
# description of a face in terms of race,gender,emotion,id
################################################################
def get_face(rac='W', gen='F', emo='HC', id='022', resize=None,
             grayscale=True, img_ref_dict={}):
    face = img_ref_dict[rac][gen][emo][id]
    return face.retrieve_img(resize=resize, crop_square=True,
                             grayscale=grayscale)
