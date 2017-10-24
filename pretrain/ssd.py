import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable


class VGGPretrain(nn.Module):
	def __init__(self):
		super(VGGPretrain, self).__init__()
		
		self.num_of_class = 114
		
		
		
		#self.VGG = self.VGG16()

		self.initNetork()
		
		
		
		#print(self.VGG)

	def initNetork(self):
	############# VGG ########################
		self.Conv2d_1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_1 = nn.ReLU(True)
		self.Conv2d_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_2 = nn.ReLU(True)
		self.MaxPool2d_1 = nn.MaxPool2d (kernel_size=2, stride=2, ceil_mode=True)
		self.Conv2d_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_3 = nn.ReLU(True)
		self.Conv2d_4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_4 = nn.ReLU(True)
		self.MaxPool2d_2 = nn.MaxPool2d (kernel_size=2, stride=2, ceil_mode=True)
		self.Conv2d_5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_5 = nn.ReLU(True)
		self.Conv2d_6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_6 = nn.ReLU(True)
		self.Conv2d_7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_7 = nn.ReLU(True)
		self.MaxPool2d_3 =  nn.MaxPool2d (kernel_size=2, stride=2, ceil_mode=True)
		self.Conv2d_8 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_8 = nn.ReLU(True)
		self.Conv2d_9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_9 = nn.ReLU(True)
		self.Conv2d_10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_10 = nn.ReLU(True)
	###################
	
		self.Conv2d_11 = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu_11 = nn.ReLU(True)
	
		self.fc_1 = nn.Linear(4096, 512)
		self.fc_2 = nn.Linear(512, self.num_of_class)
		
		
	def forward(self, x):
		x = self.Conv2d_1(x)
		x = self.relu_1(x)
		x = self.Conv2d_2(x)
		x = self.relu_2(x)
		x = self.MaxPool2d_1(x)
		x = self.Conv2d_3(x)
		x = self.relu_3(x)
		x = self.Conv2d_4(x)
		x = self.relu_4(x)
		x = self.MaxPool2d_2(x)
		x = self.Conv2d_5(x)
		x = self.relu_5(x)
		x = self.Conv2d_6(x)
		x = self.relu_6(x)
		x = self.Conv2d_7(x)
		x = self.relu_7(x)
		x = self.MaxPool2d_3(x)
		x = self.Conv2d_8(x)
		x = self.relu_8(x)
		x = self.Conv2d_9(x)
		x = self.relu_9(x)
		x = self.Conv2d_10(x)
		x = self.relu_10(x)
		
		################ END VGG #####################
		
		x = self.Conv2d_11(x)
		x = self.relu_11(x)
		
		tensor_shape = x.size()
		
		
		h = x.view(tensor_shape[0], int(tensor_shape[1] * tensor_shape[2] * tensor_shape[3]))

		h = self.fc_1(h)
		h = self.fc_2(h)

		# if not use nlllose -> use softmax here!
		return F.log_softmax(h)
		
		return h
		
		
	def VGG16(self):
		'''VGG16 layers.'''
		cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.ReLU(True)]
				in_channels = x
		return nn.Sequential(*layers)

#a = VGGPretrain()
		
class SSD300(nn.Module):
	input_size = 300

	def __init__(self):
		super(SSD300, self).__init__()

		# model
		self.base = self.VGG16()
		self.norm4 = L2Norm2d(20)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

		self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

		self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

		self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
		self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

		self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
		self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

		self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
		self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

		self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
		self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

		# multibox layer
		self.multibox = MultiBoxLayer()

	def forward(self, x):
		hs = []
		h = self.base(x)
		hs.append(self.norm4(h))  # conv4_3

		h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

		h = F.relu(self.conv5_1(h))
		h = F.relu(self.conv5_2(h))
		h = F.relu(self.conv5_3(h))
		h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

		h = F.relu(self.conv6(h))
		h = F.relu(self.conv7(h))
		hs.append(h)  # conv7

		h = F.relu(self.conv8_1(h))
		h = F.relu(self.conv8_2(h))
		hs.append(h)  # conv8_2

		h = F.relu(self.conv9_1(h))
		h = F.relu(self.conv9_2(h))
		hs.append(h)  # conv9_2

		h = F.relu(self.conv10_1(h))
		h = F.relu(self.conv10_2(h))
		hs.append(h)  # conv10_2

		h = F.relu(self.conv11_1(h))
		h = F.relu(self.conv11_2(h))
		hs.append(h)  # conv11_2

		loc_preds, conf_preds = self.multibox(hs)
		return loc_preds, conf_preds

	def VGG16(self):
		'''VGG16 layers.'''
		cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.ReLU(True)]
				in_channels = x
		return nn.Sequential(*layers)
