#####################################################
############## IMPORT ###############################
#####################################################
import argparse
import dl_gdrive
import os
import zipfile
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

parser.add_argument('--output_directory', default='./checkpoint', type=str, help='Output model directory')
parser.add_argument('--output_format', default='ckpt_%d.pth', type=str, help='Format of output model\'s name, this file must contain symbol %%d for indexing purpose [For example: ckpt_%%d.pth]')

######### MISC.
parser.add_argument('--epoch_cycle', default=50, type=int, help='For output model name format')


##########################################################
################ PRE - INITIALIZATION ####################
##########################################################
args = parser.parse_args()

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

################ NETWORK ACHITECHTURE ####################

if Network_type == 0: #SSD 300
	feature_map_sizes = (38, 19, 10, 5, 3, 1)
	steps_raw = (8, 16, 32, 64, 100, 300)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
	min_ratio = 20
	max_ratio = 90
	min_scale = 0.1

elif Network_type == 1: # SSD 400 (deprecated)
	feature_map_sizes = (50, 25, 13, 7, 5, 3, 1)
	steps_raw = (8, 16, 32, 64, 80, 133, 400)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,), (2,))
	min_ratio = 20
	max_ratio = 90
	min_scale = 0.1
		
elif Network_type == 2: # SSD 500
	feature_map_sizes = (63, 32, 16, 8, 4, 2, 1)
	steps_raw = (8, 16, 32, 64, 128, 250, 500)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,), (2,))
	min_ratio = 8
	max_ratio = 50
	min_scale = 0.03

elif Network_type == 3: # SSD 600 (Deprecated)
	feature_map_sizes = (75, 38, 19, 10, 8, 6, 4, 2) 
	steps_raw = (8, 16, 32, 60, 75, 100, 150, 300)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2, 3), (2,), (2,), (2, ))
	min_ratio = 20
	max_ratio = 90
	min_scale = 0.1

###### 

if args.resuming_model == './model/ssd.pth':
	if args.resume_mode == 'pretrain':
		print('Use are using network with resume_mode: pretrain')
		print('Be sure to specify path of pretrain model using argument: --resuming_model or default path ./model/ssd.pth will be used')
	if args.resume_mode == 'continue':
		print('Use are using network with resume_mode: pretrain')
		print('Be sure to specify path of pretrain model using argument: --resuming_model or default path ./model/ssd.pth will be used')

if '%d' not in args.output_format:
	print('--output_format param must contain %d')
	quit()

##########################################################
################ DATASET CHECKING ########################
##########################################################

print('Checking Dataset Availability ...')
if not os.path.isdir(args.train_dir):
	os.makedirs(args.train_dir)
	print ('training dataset Unavailable... ')
	print ('Downloading training Dataset ....')

	gdrive_url = 'https://drive.google.com/file/d/1jXkAHT_CfB-kHaOQI09XWdFTUlWIqfj0/view?usp=sharing'
	outpath = args.train_dir + '/train.zip'

	downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
	downloader.download()

	print ('Download Complete ....')

	zip_ref = zipfile.ZipFile(args.train_dir + '/train.zip', 'r')
	zip_ref.extractall(args.train_dir)
	zip_ref.close()

	os.remove(args.train_dir + '/train.zip')

	print ('Training dataset preparation complete ....')

#-----------------------------------------

if not os.path.isdir(args.validate_dir):
	os.makedirs(args.validate_dir)
	print ('Validation dataset Unavailable... ')
	print ('Downloading Validation Dataset ....')

	gdrive_url = 'https://drive.google.com/file/d/1jXkAHT_CfB-kHaOQI09XWdFTUlWIqfj0/view?usp=sharing'
	outpath = args.validate_dir + '/validation.zip'

	downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
	downloader.download()

	print ('Download Complete ....')

	zip_ref = zipfile.ZipFile(args.validate_dir + '/validation.zip', 'r')
	zip_ref.extractall(args.validate_dir)
	zip_ref.close()

	os.remove(args.validate_dir + '/validation.zip')

	print ('Validation dataset preparation complete ....')



###########################################
# Debugging session
# H:\anaconda_3\Install\python train.py
# Drive: https://drive.google.com/drive/folders/1Zi8SR5BhU2MxAEoF2qn1ovK1Yrl3hdMb?usp=sharing
# Train Dataset: https://drive.google.com/file/d/1jXkAHT_CfB-kHaOQI09XWdFTUlWIqfj0/view?usp=sharing
