'''Convert pretrained VGG model to SSD.

VGG model download from PyTorch model zoo: https://download.pytorch.org/models/vgg16-397923af.pth
'''
import torch

from ssd import SSD300

vgg = torch.load('./pretrain_test_off0.pth')

vgg = vgg['net']

#print(vgg.keys())


ssd = SSD300()


layer_indices = [0,1,3,4,7,8,10,11,14,15,17,18,20,21,24,25,27,28,30,31]#,24,25,27,28,30,31]

for layer_idx in layer_indices:
    ssd.base[layer_idx].weight.data = vgg['features.%d.weight' % layer_idx]
    ssd.base[layer_idx].bias.data = vgg['features.%d.bias' % layer_idx]

# [24,26,28]
ssd.conv5_1.weight.data = vgg['features.34.weight']
ssd.conv5_1.bias.data = vgg['features.34.bias']
ssd.conv5_2.weight.data = vgg['features.37.weight']
ssd.conv5_2.bias.data = vgg['features.37.bias']
ssd.conv5_3.weight.data = vgg['features.40.weight']
ssd.conv5_3.bias.data = vgg['features.40.bias']

ssd.norm5_1.weight.data = vgg['features.35.weight']
ssd.norm5_1.bias.data = vgg['features.35.bias']
ssd.norm5_2.weight.data = vgg['features.38.weight']
ssd.norm5_2.bias.data = vgg['features.38.bias']
ssd.norm5_3.weight.data = vgg['features.41.weight']
ssd.norm5_3.bias.data = vgg['features.41.bias']


torch.save(ssd.state_dict(), './checkpoint/ssd.pth')