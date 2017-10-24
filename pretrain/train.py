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
import pdb
import matplotlib.pyplot as plt

##################### PARAMETER DIFINITION #############################
p_batch_size = 2
Continue_training = False 
lr = 0.001
epoch_count = 3
########################################################################




#use_cuda = torch.cuda.is_available()
train_ite = 0
test_ite = 0
use_cuda = False
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
all_loss = []
all_testloss=[]


# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='./zipdataset/train/', list_file='./pretraindataconfig/train_path.txt', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=p_batch_size, shuffle=True)

testset = ListDataset(root='./zipdataset/test/', list_file='./pretraindataconfig/test_path.txt', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=p_batch_size, shuffle=False)


# Model
net = VGGPretrain()
#state = {
#	'net': net.state_dict(),
#	'loss': 1,
#	'epoch': 1,
#}
#
#torch.save(state, './checkpoint/temppp.pth')
if use_cuda:
	net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
	net.cuda()
	cudnn.benchmark = True
	
if Continue_training:
	print('==> Resuming from checkpoint..')
	checkpoint = torch.load('./checkpoint/pretrain1.pth')
#	for e in checkpoint:
#		print(e)
#		pdb.set_trace()
	a = net.load_state_dict(checkpoint['net'])
#	for e, v in enumerate(checkpoint['net']):
#		print(v)
#		pdb.set_trace()
	best_loss = checkpoint['loss']
	start_epoch = checkpoint['epoch']
else:
	# Convert from pretrained VGG model.
	pass
	#net.load_state_dict(torch.load('./model/ssd.pth'))

criterion = nn.NLLLoss()
m = nn.LogSoftmax()



optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)



# Training
def train(epoch):
#	all_loss = []
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	br = 0
	global train_ite, all_loss
	if epoch%1==0:
		all_loss = []
	for batch_idx, (images, label) in enumerate(trainloader):
		if use_cuda:
			images = images.cuda()
			label = label.cuda()

		images = Variable(images)
		label = Variable(label)

#		pdb.set_trace()
		optimizer.zero_grad()
		pred = net(images)
#		pdb.set_trace()
		label = label.view(p_batch_size)
		
		loss = criterion(pred, label)
		
		
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
#		pdb.set_trace()
		plt.ion()
		if train_ite%1==0:
			all_loss.append(train_loss/(batch_idx+1))
			
			plt.clf()
			plt.plot(all_loss)
			plt.draw()
			plt.savefig('figures/tmp_train.png')
#		
		print('[E %d, I %d]: %.3f, %.3f' % (epoch,batch_idx, loss.data[0], train_loss/(batch_idx+1)))
#		print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
		train_ite+=1
		print(train_ite)

def test(epoch):
#	all_testloss = []
	global test_ite, all_testloss
	if epoch%1==0:
		all_testloss = []
	net.eval()
	test_loss = 0
	for batch_idx, (images, label) in enumerate(testloader):
		if use_cuda:
			images = images.cuda()
			label = label.cuda()

		images = Variable(images, volatile=True)
		label = Variable(label)

		pred = net(images)
		
		label = label.view(p_batch_size)
		
		loss = criterion(pred, label)
		test_loss += loss.data[0]
		plt.ion()
		if test_ite%10==9:
			all_testloss.append(test_loss/(batch_idx+1))
			plt.clf()
			plt.plot(all_testloss)
			plt.savefig('figures/tmp_test.png')
#		print('%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1)))
		print('[E %d, I %d]: %.3f, %.3f' % (epoch,batch_idx, loss.data[0], test_loss/(batch_idx+1)))
		test_ite+=1
		print(test_ite)
		
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
	
