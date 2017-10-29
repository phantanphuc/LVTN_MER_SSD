'''Load image/class/box from a annotation file.

The annotation file is organized as:
    image_name #obj xmin ymin xmax ymax class_index ..
'''
from __future__ import print_function
import cv2
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb

from PIL import Image, ImageOps


class ListDataset(data.Dataset):
    img_size = 128

    def __init__(self, root, list_file, transform):

        self.root = root

        self.fnames = []
        self.labels = []

        self.transform = transform
        
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        #### format: name label

        
        
        for line in lines:
            label = []
            

            splited = line.strip().split()
            self.fnames.append(splited[0])

            cur_label = int(splited[1])
            label.append(int(cur_label))

            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):

        # Load image and bbox locations.
        fname = self.fnames[idx]

        #print(os.path.join(self.root, fname))
        #quit()

        img = Image.open(os.path.join(self.root, fname))
#        pdb.set_trace()
       

        labels = self.labels[idx]
        img = img.resize((self.img_size,self.img_size))
        img = self.transform(img)

        #cv2.imshow('aaa', img.numpy()[0])
        #cv2.waitKey()



        # Encode loc & conf targets.
        #loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, labels#, conf_target


    def __len__(self):
        return self.num_samples
