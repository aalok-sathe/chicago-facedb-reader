from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import os
import shutil
import json
import random
import numpy as np

from . import preparation
from . import faces

home = os.path.expanduser('~')
inst = os.path.join(home, ".cfd-faces")

try:
    bool, cached = preparation.is_prepared(home=home, inst=inst)
    assert bool
except AssertionError:
    preparation.prepare(home=home, inst=inst)

def clean():
    confirmation = input("Clean installation? You will have to reindex faces"
                       + " on the next run. (y/n): ").lower()
    if confirmation == 'y':
        try:
            shutil.rmtree(inst)
            print("Installation cleaned. Exiting.")
            raise SystemExit
        except IOError:
            print("Failed to uninstall. Please delete manually from", inst)

def load_data(grayscale=True, train_proportion=.9, resize=None):
    """Method for use in other scripts and/or modules
    to produce DB data in a systematic manner, split into
    a training set and a test set (similar to the keras-MNIST method)"""
    # load json objects storing face references
    with open(os.path.join(inst,'images.json'),'r') as imgjson:
        img_ref_dict = json.load(imgjson)
    with open(os.path.join(inst,'indexed.json'),'r') as indjson:
        indexed_faces = json.load(indjson)

    all = list(indexed_faces)
    random.shuffle(all)
    train_set = all[:int(len(all)*train_proportion)]
    test_set = all[int(len(all)*train_proportion):]
    returnable = [
        [np.array([], dtype=np.float32),
         np.array([], dtype=np.float32)], # train
        [np.array([], dtype=np.float32),
         np.array([], dtype=np.float32)], # test
    ]
    # iterate over entries in train and test sets and add labels to array
    print("\nPreparing train set using", train_proportion, "of data")
    for item in progressbar(train_set, redirect_stdout=True):
        entry = dict(zip(['rac', 'gen', 'emo', 'id'], item.split()))
        returnable[0][0] = np.append(returnable[0][0],
                                     [faces.get_face(**entry,
                                      grayscale=grayscale, resize=resize)])
        returnable[0][1] = np.append(returnable[0][1],
                                     np.array([
                                        Race[entry['rac']].value,
                                        Gender[entry['gen']].value,
                                        Emotion[entry['emo']].value,
                                     ], dtype=np.uint8))
        if resize == None:
            resize = faces.get_face(**entry).shape[:2]

    if grayscale: channels = 1
    else: channels = 3

    print("\nPreparing test set using", 1-train_proportion, "of data")
    for item in progressbar(test_set, redirect_stdout=True):
        entry = dict(zip(['rac', 'gen', 'emo', 'id'], item.split()))
        returnable[1][0] = np.append(returnable[1][0],
                                     [faces.get_face(**entry,
                                      grayscale=grayscale, resize=resize)])
        returnable[1][1] = np.append(returnable[1][1],
                                     np.array([
                                        Race[entry['rac']].value,
                                        Gender[entry['gen']].value,
                                        Emotion[entry['emo']].value,
                                     ], dtype=np.uint8))

    for inputset in (0,1):
        returnable[inputset][1] = returnable[inputset][1].reshape(-1,3,1)

    for inputset in (0,1):
        returnable[inputset][0] = np.expand_dims(returnable[inputset][0],
                                   axis=3).reshape(returnable[0][1].shape[0],
                                                   *resize, channels)


    return returnable
