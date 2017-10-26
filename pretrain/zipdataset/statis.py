import os
from os import walk
from shutil import copyfile
import pdb

filename = 'test_path.txt'

#88

def statis():
	for (dirpath, dirnames, filenames) in walk('./'):
		if dirpath != './':
			print('=======')
			print(dirpath)
			print(len(filenames))

def generateLabel():
	write_file = open('label.txt', 'w')
	content = ''
	idx = 0
	for (dirpath, dirnames, filenames) in walk('./Train'):
#		pdb.set_trace()
#		if dirpath == './':
		for dirs in dirnames:
			content += dirs + ' ' + str(idx) + '\n'
			idx += 1
	write_file.write(content)
	write_file.close()

def loadLabel():
	
	dictionary = {}
	idx = 0

	with open('label.txt') as f:
		content = f.readlines()
		for symbol in content:
			symbol = symbol.replace('\n','')

			split = symbol.split(' ')

			dictionary[split[0]] = int(split[1])
#	pdb.set_trace()
	return dictionary

def generateFilePath():

	dictionary = loadLabel()

	write_file = open(filename, 'w')
	content = ''
	for (dirpath, dirnames, filenames) in walk('./'):
		
		if dirpath != './' and dirpath != './Test':
			print(dirpath)
#			pdb.set_trace()
			current_idx = dictionary[dirpath[7:]]
			for fname in filenames:
				content += dirpath[7:] + '/' + fname + ' ' + str(current_idx) + '\n'
				print(fname)

	write_file.write(content)

	write_file.close()

def splitDataset(dest, test_size = 50):
	for (dirpath, dirnames, filenames) in walk('./'):
		if dirpath != './':
			for i in range(50):
				#print(dest + '\\' + dirpath[2:]  + '\\' + filenames[i])
				copyfile(dirpath + '\\' + filenames[i], dest + '\\' + dirpath[2:]  + '\\' + filenames[i])
				os.remove(dirpath + '\\' + filenames[i])
				#quit()


#generateLabel()
#loadLabel()
#statis()
generateFilePath()

#splitDataset('E:\LV\img\Test')