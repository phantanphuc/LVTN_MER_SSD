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

from ssd import SSD300
from datagen import ListDataset
from multibox_loss import MultiBoxLoss

from torch.autograd import Variable



############ PARAM #########################3333

use_cuda = False#torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
epoch_count = 10

learning_rate = 0.001
resume = True

batch_size = 1
####################################################

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='./dataset/Exp_Test/Exp_Test_BKNgoc', list_file='./dataset/ssd_test_BKN_debug.txt', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

testset = ListDataset(root='./dataset/Exp_Test/Exp_Test_BKNgoc', list_file='./dataset/ssd_test_BKN_debug.txt', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True)


net = SSD300()

if use_cuda:
    if resume:
        pass
        #net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
    net.cuda()
    cudnn.benchmark = True

if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./model/ssdtrain0511_11.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    #print(torch.load('./model/ssd.pth').keys())

    net.load_state_dict(torch.load('./model/ssd.pth'))


criterion = MultiBoxLoss()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        images = Variable(images)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        optimizer.zero_grad()


        loc_preds, conf_preds = net(images)

        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)

        try:
            loss.backward()
        except:
            print('err')
        optimizer.step()

        train_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))

        quit()

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        images = Variable(images, volatile=True)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        test_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint.
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
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch + epoch_count):
    train(epoch)
    test(epoch)