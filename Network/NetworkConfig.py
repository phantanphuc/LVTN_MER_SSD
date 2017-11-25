#####################################################
############## IMPORT ###############################
#####################################################
import argparse


#####################################################
############## PARAMETER ############################
#####################################################

###
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
###
######### HYPER PARAM
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--decay', default=1e-4, type=float, help='decay')
parser.add_argument('--use_cuda', default=False, type=bool, help='Use CUDA for training')
parser.add_argument('--epoch_count', default=10, type=int, help='Number of training epoch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--resume_mode', default='none', type=str, help='Continue training mode: \'none\': From nothing,\'pretrain\': From pretrain model, \'continue\': Continue from SSD Model ')

######### Core Component
parser.add_argument('--using_python_2', default=False, type=bool, help='Current python version')
parser.add_argument('--class_count', default=107, type=int, help='Number of classes')
parser.add_argument('--network', default='SSD500', type=str, help='network type: \'SSD300\': use original SSD300, \'SSD500\': Improved version ')

######### PATH 
parser.add_argument('--train_dir', default='./dataset/train', type=str, help='training set directory')
parser.add_argument('--train_meta', default='./metafile/train.txt', type=str, help='training set metafile location')

parser.add_argument('--validate_dir', default='./dataset/train', type=str, help='validation set directory')
parser.add_argument('--validate_meta', default='./metafile/train.txt', type=str, help='validateion set metafile location')

parser.add_argument('--resuming_model', default='./model/ssd.pth', type=str, help='Model to load (Only valid for resume_mode: pretrain and continue)')

######### MISC.

##########################################################
################ PRE - INITIALIZATION ####################
##########################################################
args = parser.parse_args()
###### 

if args.resuming_model == './model/ssd.pth':
	if args.resume_mode == 'pretrain':
		print('Use are using network with resume_mode: pretrain')
		print('Be sure to specify path of pretrain model using argument: --resuming_model or default path ./model/ssd.pth will be used')
	if args.resume_mode == 'continue':
		print('Use are using network with resume_mode: pretrain')
		print('Be sure to specify path of pretrain model using argument: --resuming_model or default path ./model/ssd.pth will be used')


##############################
#### NETWORK TYPE
# 0: SSD 300
# 1:
# 2: SSD 500 improved
InputImgSize = 300
Network_type = 0
if args.network == 'SSD400':
	Network_type = 1
	InputImgSize = 400
elif args.network == 'SSD500':
	Network_type = 2
	InputImgSize = 500
elif args.network == 'SSD600':
	Network_type = 3
	InputImgSize = 600


