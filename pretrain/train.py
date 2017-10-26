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
from ssd_new import vgg16_bn
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
epoch_count = 30
########################################################################




#use_cuda = torch.cuda.is_available()
train_ite = 0
test_ite = 0
use_cuda = False
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch



# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='./zipdataset/train/', list_file='./pretraindataconfig/train_patha.txt', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=p_batch_size, shuffle=True)

testset = ListDataset(root='./zipdataset/test/', list_file='./pretraindataconfig/train_patha.txt', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=p_batch_size, shuffle=False)


# Model


if False:
	net = VGGPretrain()
else:
	net = vgg16_bn(False)
	print(vgg16_bn(False))

if use_cuda:
	net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
	net.cuda()
	cudnn.benchmark = True
	
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

criterion = nn.CrossEntropyLoss()
#sm = nn.LogSoftmax()



optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


def try_print(print_flag = True):
	params = [p for p in list(net.parameters()) if p.requires_grad==True]
	for p in params:
		print(p)
		p_grad = p.grad 
		pdb.set_trace()
		try:
			if print_flag:
				print ('exist')
				print (type(p_grad))
				print (p_grad.data.numpy().shape)
			else:
				print (p_grad.data.numpy())
					
		except:
			if print_flag:
				print ('non - exist')
				pass


# Training
def train(epoch):

	all_loss = []
	all_testloss=[]

	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	br = 0
	global train_ite
	for batch_idx, (images, label) in enumerate(trainloader):
		if use_cuda:
			images = images.cuda()
			label = label.cuda()

		images = Variable(images)
		label = Variable(label)


		optimizer.zero_grad()
		pred = net(images)
#		pdb.set_trace()
		label = label.view(p_batch_size)
		
		loss = criterion(pred, label)
		
		#try_print()
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
#		pdb.set_trace()
		if train_ite%1==2:
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
	all_loss = []
	all_testloss=[]
	global test_ite

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
		if test_ite%1==3:
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
#	try_print()
	train(epoch)
	test(epoch)
	
