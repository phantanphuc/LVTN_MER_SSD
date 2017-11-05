
count = 0

err_list = []

with open('./a.txt') as f:
	content = f.readlines()
	for i in range(len(content)):
		if content[i] == 'err\n':
			count = count + 1
			err_list.append(content[i - 1].replace('\n',''))

print(count)

print(err_list)

out_content = ''

with open('./ssd_validate.txt') as f:
	content = f.readlines()
	for i in content:
		is_err = False
		for c in err_list:
			if c in i:
				is_err = True
		if not is_err:
			out_content = out_content + i

with open('./ssd_validate2.txt', 'w') as f:
	f.write(out_content)
	f.close()