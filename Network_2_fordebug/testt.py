import math
import os
import shutil
import stat
import subprocess
import sys

min_dim = 300
mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
# in percent %
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))

print(step)
print('------------')

sizes = []
for ratio in range(min_ratio, max_ratio + 1 + step, step):
	print(ratio)

	sizes.append(min_dim * ratio / 100.)
sizes = [min_dim * 10 / 100.] + sizes



print(sizes)
