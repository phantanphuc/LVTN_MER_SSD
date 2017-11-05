import numpy as np

FILE = 'ssd_validate_strip.txt'
#FILE = 'test.txt'

dictionary = {}
res_dict = []

with open('./label.txt') as f:
	content = f.readlines()
	for symbol in content:
		symbol = symbol.replace('\n','')

		split = symbol.split(' ')

		dictionary[split[0]] = int(split[1])
		res_dict.append(split[0])



SYM_COUNT = np.zeros(len(res_dict))
OCC_IN_IMG_COUNT = np.zeros(len(res_dict))

mask = [False] * len(res_dict)

with open(FILE) as f:
	data = f.readlines()
	for line in data:
		mask = [False] * len(res_dict)

		list_data = line.replace('\n', '').split(' ')

		print(list_data)

		numofBB = int(list_data[1])
		idx = 2
		#print(numofBB)

		for i in range(numofBB):
			SYM_COUNT[int(list_data[i * 5 + 2 + 4])] += 1
			mask[int(list_data[i * 5 + 2 + 4])] = True

		for i in range(len(mask)):
			if mask[i] == True:
				OCC_IN_IMG_COUNT[i] += 1

print(SYM_COUNT)
print(OCC_IN_IMG_COUNT)

content = 'SYM, COUNT, OCCUR IN IMG\n'
with open('res.csv', 'w') as f:
	for i in range(len(res_dict)):
		content = content + str(res_dict[i]) + ',' + str(SYM_COUNT[i]) + ',' + str(OCC_IN_IMG_COUNT[i]) + '\n'
	f.write(content)
	f.close()