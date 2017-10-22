#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 00:51:55 2017

@author: ngocbui
"""

#bi loi train loader neu so file du khong khop voi batch size
import numpy as np
import pdb

from scipy import ndimage
from scipy import misc
#a = np.array([[0, 0, 0, 0, 0, 0, 0],
#              [0, 1, 1, 0, 0, 0, 0],
#              [1, 1, 1, 1, 0, 0, 0],
#              [0, 0, 1, 1, 1, 0, 0],
#              [0, 0, 1, 1, 1, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 1, 1, 1, 0, 0]])  # Second object here
# Label objects
a = misc.imread('./img1.jpg')
labeled_image, num_features = ndimage.label(a)
# Find the location of all objects
objs = ndimage.find_objects(labeled_image)
#pdb.set_trace()
# Get the height and width
measurements = []
for ob in objs:
	measurements.append((int(ob[0].stop - ob[0].start), int(ob[1].stop - ob[1].start)))
pdb.set_trace()