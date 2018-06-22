#! /bin/env/ python3

from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import os # for filesystem access
import cv2 # opencv library for image processing
import typing # for function annotations
import json # to dump objects in the json format (as opposed to pickled binary)
import xlrd,csv # to read the CFD data codebook and output it to a csv file
from progressbar import progressbar # to display progress during iteration
from collections import defaultdict # to use during indexing faces

from . import faces

def is_prepared(home=None, inst=None) -> bool:
    """Checks to see if the installation directory exists and is well"""
    bool = os.path.isdir(inst)
    cached = os.path.isdir(os.path.join(inst, 'images'))
    for file in {'cfd_stats.csv','images.json'}:
        bool &= os.path.isfile(os.path.join(inst, file))
    if not bool: print ("Installation is pending, broken, or incomplete.")
    return bool, cached

def prepare(home=None, inst=None, cfddir=None) -> None:
    """Prepares installation directory, processes spreadsheet data, and
       indexes faces in the CFD images directory and dumps them to .json"""
    # make installation directory in user's home
    try:
        os.mkdir(inst)
        os.mkdir(os.path.join(inst,'images'))
    except PermissionError:
        print("Error, could not make directory", inst,": insufficient access")
        raise SystemExit

    # retrieve user's downloaded copy of the Chicago Face DB
    if cfddir == None:
        cfddir = input("Please input the full path to the CFD directory that "
         + "directly contains the images directory and the data spreadsheet.\n")
    bool = os.path.isdir(cfddir)

    # see if the user has provided a correct dir that is properly structured
    try:
        cfddirls = [os.path.join(cfddir, x) for x in os.listdir(cfddir)]
        imgsdir = list(filter(lambda x: os.path.isdir(x), cfddirls))
        excelsh = list(filter(lambda x: x.endswith('.xlsx'), cfddirls))
    except FileNotFoundError:
        print("Error finding and accessing supplied directory at", cfddir)
        raise SystemExit

    # evaluate validity of expected files in the directory
    bool &= len(imgsdir) == 1 & len(excelsh) == 1
    # assert condition that all checks are met, exit with error message if not
    assert bool, "Invalid path supplied. Please re-run and supply correct path"\
               + "\nThe path must directly contain the image directory, and "\
               + "the data spreadsheet. If you haven't, please download the "\
               + "Chicago Face Database and extract it to a convenient location"
    imgsdir = imgsdir[0]
    excelsh = excelsh[0]

    # process xlsx file and extract first sheet into instdir
    try:
        excelsh = xlrd.open_workbook(excelsh)
    except FileNotFoundError:
        print("Error, could not open CFD face data spreadsheet")
        raise SystemExit
    data_sheet = excelsh.sheet_by_index(0)
    # use the csv module to output into csv
    with open(os.path.join(inst,'data.csv'), 'w') as csvout:
        csvwriter = csv.writer(csvout, quoting=csv.QUOTE_ALL)
        for row in range(data_sheet.nrows):
            csvwriter.writerow(data_sheet.row_values(row))

    img_containers = [os.path.join(imgsdir, name) for name in os.listdir(imgsdir) if '-' in name]

    # gather options for index_faces operation
    print("Beginning to index faces.")
    crop_square = True
    if input("Crop the faces square? (y/n) ").lower() == 'n':
        crop_square = False
    cache = True
    if input("Cache image files in install directory? (y/n): ").lower() == 'n':
        cache = False
    resize = input("Resize images to custom dimentions? Input 'height width',"
                 + " e.g. (default): 32 32. 'n' for NO.\n").lower()
    if resize == 'n': resize = None
    elif not len(resize): resize = (32,32)
    else: resize = tuple(resize.split())

    (img_ref_dict,
    indexed_faces) = faces.index_faces(imgsdir=imgsdir, inst=inst,
                                       img_containers=img_containers,
                                       cache=cache, resize=resize,
                                       crop_square=crop_square)
