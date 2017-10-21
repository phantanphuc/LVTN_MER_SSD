#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:02:25 2017

@author: ngocbui
"""

import os
from os import walk
import re
import collections
import pdb
def readSymbolfile(path):
#	pdb.set_trace()
	assert(os.path.exists(path))
	with open(path, 'r') as f:
		return f.read().replace("\n", " ").split()
	
def buildVocab(path):
	
	data = readSymbolfile(path)
	counter = collections.Counter(data)
	#print(counter)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	#print(count_pairs)
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	id_to_word = dict((v, k) for k, v in word_to_id.items())
	#print(len(word_to_id))
	#print(id_to_word)
	#train = _file_to_word_ids(truth, word_to_id)
	#print(train)
	return word_to_id, id_to_word