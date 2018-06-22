#! /bin/env/ python3

import os
import cv2
import pickle
import random
import numpy as np
import enum
from collections import defaultdict
from progressbar import progressbar

def supply_dict(key = 'genders',
                 legend = {'genders' : 0, 'emotions' : 1}):
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

class face_provider:

    data_path = '../data/cfd2.0.3/images'
    races = {'A', 'B', 'L', 'W'}
    genders = {'F', 'M'}
    emotions = {'A', 'F', 'HC', 'HO', 'N'}

    def __init__(self, path=data_path):
        self.image_containers = os.listdir(os.path.abspath(path))
        self.images = {
            'A' : supply_dict(), 'W' : supply_dict(),
            'B' : supply_dict(), 'L' : supply_dict(),
        }
        self.indexed_faces = set()

    def load_from_pickle(self, path="pickled/"):
        with open(os.path.join(path, "images.pickle"), 'rb') as file:
            self.images = pickle.load(file)
        with open(os.path.join(path, "image_containers.pickle"), 'rb') as file:
            self.image_containers = pickle.load(file)
        with open(os.path.join(path, "indexed_faces.pickle"), 'rb') as file:
            self.indexed_faces = pickle.load(file)

    def dump_to_pickle(self, path="pickled/"):
        with open(os.path.join(path, "images.pickle"), 'wb') as file:
            pickle.dump(self.images, file)
        with open(os.path.join(path, "image_containers.pickle"), 'wb') as file:
            pickle.dump(self.image_containers, file)
        with open(os.path.join(path, "indexed_faces.pickle"), 'wb') as file:
            pickle.dump(self.indexed_faces, file)

    def index_faces(self,
                     path = data_path):
                     # races = {'A', 'W', 'B', 'L'},
                     # genders = {'M', 'F'},
                     # emotions = {'A', 'F', 'N', 'HO', 'HC'}):

        print("Indexing faces")

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
                # Export processed image to directory, if needed elsewhere
                # cv2.imwrite("../data/processed_dump/%s"%filename, self.images[rac][gen][emo][id])

    def crop_square(self, rac='W', gen='F', emo='HC', id='022'):
        """Crop image to a square with dimension that is lowest
        of height and width. Whichever one of those dimensions is
        greater is reduced to the newly determined dimension of
        the square, using half-delta reduction from two ends"""
        img = self.images[rac][gen][emo][id]
        h,w = img.shape[0:2]
        d = 0.5 * abs(h-w)
        # print(h,w,d)
        cropped_img = img[int((h>w)*d):int(h-(h>w)*d),
                          int((w>h)*d):int(w-(w>h)*d)]
        # self.images[rac][gen][emo][id] = cropped_img
        return cropped_img

    def resize(self, rac='W', gen='F', emo='HC', id='022', resize=(28,28)):
        """Resize image to supplied dimensions"""
        img = self.images[rac][gen][emo][id]
        if resize!=img.shape[0:2]:
            img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
        return img
        # self.images[rac][gen][emo][id] = img

    def get_face(self, grayscale=True, rac='W', gen='F', emo='HC', id='022',
                  resize=(28,28)):
        """Return a singular face image object from DB"""
        # img = pickle.load(open(self.images[rac][gen][emo][id]), 'rb') # WIP
        img = self.resize(rac, gen, emo, id, resize=resize)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (*img.shape[0:2], 1))
            # print(img.shape)
        # else:
        #     print(grayscale)
        return img

    def list_faces(self, rac=None, gen=None, emo=None):
        """Return a list of faces of all persons matching the provided
        race, gender, and emotion"""
        races = [rac]; genders = [gen]; emotions = [emo]
        if rac == None: races = self.races
        if gen == None: genders = self.genders
        if emo == None: emotions = self.emotions
        l = [img
              for rac in races
               for gen in genders
                for emo in emotions
                 for img in self.images.get(rac,{}).get(gen,{}).get(emo,set())]
        return l

    def load_data(self, grayscale=True, train_proportion=.9, resize=(28,28)):
        """Method for use in other scripts and/or modules
        to produce DB data in a systematic manner, split into
        a training set and a test set (similar to the keras-MNIST method)"""
        # print(self.indexed_faces)
        all = list(self.indexed_faces)
        random.shuffle(all)
        train_set = all[:int(len(all)*train_proportion)]
        test_set = all[int(len(all)*train_proportion):]
        returnable = [
            [np.array([], dtype=np.float32),
             np.array([], dtype=np.float32)], # Train
            [np.array([], dtype=np.float32),
             np.array([], dtype=np.float32)], # Test
        ]
        # Iterate over entries in train and test sets and add labels to array
        print("\nPreparing train set using", train_proportion, "of data")
        for item in progressbar(train_set, redirect_stdout=True):
            entry = dict(zip(['rac', 'gen', 'emo', 'id'], item.split()))
            returnable[0][0] = np.append(returnable[0][0], [self.get_face(**entry, grayscale=grayscale, resize=resize)])
            returnable[0][1] = np.append(returnable[0][1],
             # [Gender[entry['gen']].value])
             np.array([
                Race[entry['rac']].value,
                Gender[entry['gen']].value,
                Emotion[entry['emo']].value,
             ], dtype=np.uint8))
        print("\nPreparing test set using", 1-train_proportion, "of data")
        for item in progressbar(test_set, redirect_stdout=True):
            entry = dict(zip(['rac', 'gen', 'emo', 'id'], item.split()))
            returnable[1][0] = np.append(returnable[1][0], [self.get_face(**entry, grayscale=grayscale, resize=resize)])
            returnable[1][1] = np.append(returnable[1][1],
             # [Gender[entry['gen']].value])
             np.array([
                Race[entry['rac']].value,
                Gender[entry['gen']].value,
                Emotion[entry['emo']].value,
             ], dtype=np.uint8))

        return returnable

if __name__ == '__main__':
    fp = face_provider()
    fp.index_faces()
    fp.dump_to_pickle()

    print(fp.list_faces('W','F','HC'))
