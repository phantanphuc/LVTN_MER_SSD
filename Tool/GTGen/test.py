dictionary = {}

with open('./label.txt') as f:
	content = f.readlines()
	for symbol in content:
		symbol = symbol.replace('\n','')
		split = symbol.split(' ')

		dictionary[split[0]] = int(split[1])

#print(dictionary.keys()[1])

for i in dictionary.keys():
	print(i)
