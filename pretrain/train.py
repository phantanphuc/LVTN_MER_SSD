from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from ssd import VGGPretrain
#from utils import progress_bar
from datagen import ListDataset

from torch.autograd import Variable

import cv2

##################### PARAMETER DIFINITION #############################
p_batch_size = 2
Continue_training = False
lr = 0.001
epoch_count = 10
########################################################################




#use_cuda = torch.cuda.is_available()
use_cuda = False
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='./dataset/train/', list_file='./pretraindataconfig/filepath_train.sources.txt', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=p_batch_size, shuffle=True)

testset = ListDataset(root='./dataset/test/', list_file='./pretraindataconfig/filepath_test.sources.txt', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=p_batch_size, shuffle=False)


# Model
net = VGGPretrain()
if Continue_training:
	print('==> Resuming from checkpoint..')
	checkpoint = torch.load('./checkpoint/ckpt0.pth')
	net.load_state_dict(checkpoint['net'])
	best_loss = checkpoint['loss']
	start_epoch = checkpoint['epoch']
else:
	# Convert from pretrained VGG model.
	pass
	#net.load_state_dict(torch.load('./model/ssd.pth'))

criterion = nn.NLLLoss()
m = nn.LogSoftmax()

if use_cuda:
	net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
	net.cuda()
	cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)



# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	br = 0
	for batch_idx, (images, label) in enumerate(trainloader):
		if use_cuda:
			images = images.cuda()
			label = label.cuda()

		images = Variable(images)
		label = Variable(label)

		optimizer.zero_grad()
		pred = net(images)
		

		label = label.view(p_batch_size)
		
		#quit()
		
		loss = criterion(pred, label)
		
		
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
		
		

def test(epoch):

	net.eval()
	test_loss = 0
	for batch_idx, (images, label) in enumerate(trainloader):
		if use_cuda:
			images = images.cuda()
			label = label.cuda()

		images = Variable(images, volatile=True)
		label = Variable(label)

		pred = net(images)
		
		label = label.view(p_batch_size)
		
		loss = criterion(pred, label)
		test_loss += loss.data[0]
		print('%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

	global best_loss
	test_loss /= len(testloader)
	if test_loss < best_loss:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'loss': test_loss,
			'epoch': epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/pretrain' + str(epoch) + '.pth')
		best_loss = test_loss



for epoch in range(start_epoch, start_epoch + epoch_count):
	train(epoch)
	test(epoch)
	
