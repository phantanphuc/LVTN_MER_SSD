import tkinter
from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import ttk
import tkinter.messagebox
import os

class MainWindow:
	def __init__(self):
		print ('initializing...')
		
		self.mainHandle = tkinter.Tk()
		#self.mainHandle.configure(background='black')
		self.mainHandle.title('Groundtruth Generator')
		self.mainHandle.geometry("600x400")
		
		self.camvas_border = 0 
		self.BB_in_Canvas = []
		self.Text_in_Canvas = []
		self.file_list = []

		self.img_idx = 0

		self.file_path = ''
		self.initGUI()
		self.initKeyBinding()
		
		self.dictionary = {}
		self.dictindex = []
		with open('./label.txt') as f:
			content = f.readlines()
			for symbol in content:
				symbol = symbol.replace('\n','')

				split = symbol.split(' ')

				self.dictionary[split[0]] = int(split[1])

		for i in self.dictionary.keys():
			self.dictindex.append(i)

		#########################3

	def initGUI(self):
	
		############# BROWSE #############################
	
		file_frame = tkinter.Frame(master=self.mainHandle, bd = 10)
		file_frame.pack(fill=tkinter.BOTH)
		
		self.f1Entry = tkinter.Entry(file_frame, bd = 5, width= 50)
		self.f1Entry.pack(side = tkinter.LEFT)
		
		f1Browse = tkinter.Button(file_frame, text = "Browse Folder", width=15, command=self.open_callback)
		f1Browse.pack(side = tkinter.LEFT)
		
		LoadBrowse = tkinter.Button(file_frame, text = "Load", width=15, command=self.loadFile)
		LoadBrowse.pack(side = tkinter.LEFT)
		
		##### Choose

		
		##### IMG
		
		img_frame = tkinter.Frame(master=self.mainHandle, bd = 0)
		img_frame.pack(fill=tkinter.BOTH)
	
		
		#self.img = ImageTk.PhotoImage(Image.open('crop.jpg'))
		self.img = ImageTk.PhotoImage(Image.new('RGB', (512, 256)))
		
		self.img_holder = tkinter.Label(img_frame, image = self.img, borderwidth=10, relief="raised")
		#self.img_holder.pack(side = tkinter.LEFT)

		self.img_canvas = tkinter.Canvas(img_frame, height=256, width=512)#, borderwidth=self.camvas_border, relief="raised")
		self.img_canvas.pack(side = tkinter.LEFT)
		self.img_canvas.bind("<Button-1>", self.mouseEventDown)
		self.img_canvas.bind("<ButtonRelease-1>", self.mouseEventUp)
		self.img_canvas.bind("<B1-Motion>", self.mouseEventMove)
		# tut: http://effbot.org/tkinterbook/tkinter-events-and-bindings.htm
		self.image_on_canvas = self.img_canvas.create_image(self.camvas_border, self.camvas_border, anchor = tkinter.NW, image = self.img)
		

		
	def mouseEventDown(self, event):
		pass
	
	def mouseEventUp(self, event):
		pass

	def mouseEventMove(self, event):
		pass

	def undo(self, event=None):
		pass



	def delBBList(self):
		for rect in self.BB_in_Canvas:
			self.img_canvas.delete(rect)
		del self.BB_in_Canvas
		self.BB_in_Canvas = []


		for rect in self.Text_in_Canvas:
			self.img_canvas.delete(rect)
		del self.Text_in_Canvas
		self.Text_in_Canvas = []

	def showIMG(self, idx):
		self.delBBList()
		self.img = ImageTk.PhotoImage(Image.open(self.file_path + self.file_list[idx][0]))
		self.img_canvas.itemconfig(self.image_on_canvas, image = self.img)
		for i in range(len(self.file_list[idx][1])):
			rect = self.showIMGBB(idx, i)
			self.BB_in_Canvas.append(rect)

			self.Text_in_Canvas.append(self.img_canvas.create_text(self.file_list[idx][1][i][0],self.file_list[idx][1][i][1],fill="darkblue", text=self.dictindex[self.file_list[idx][1][i][4]]))

	def showIMGBB(self, idx, BB_idx):
		rect = self.img_canvas.create_rectangle(self.file_list[idx][1][BB_idx][0], self.file_list[idx][1][BB_idx][1], self.file_list[idx][1][BB_idx][2], self.file_list[idx][1][BB_idx][3], outline='red')
		return rect

	def loadFile(self):
		file_path = filedialog.askopenfilename(initialdir='.')
		with open(file_path) as f:
			linecontent = f.readlines()

			for line in linecontent:
				fname = line[:line.find(' ')]
				data = line[line.find(' ') + 1:].replace('\n','')
				data_list = data.split(' ')

				BB_count = int(data_list[0])
				data_list = data_list[1:]

				BB_list = []

				for i in range(BB_count):
					idx = i * 5
					BB_list.append((int(data_list[idx]), int(data_list[idx + 1]), int(data_list[idx + 2]), int(data_list[idx + 3]), int(data_list[idx + 4])))

				#print(data)
				self.file_list.append((fname, BB_list))
				
		print(self.file_list)
		self.showIMG(0)

	def initKeyBinding(self):
		#self.mainHandle.bind('<Escape>', self.close)
		self.mainHandle.bind('<Control-z>', self.undo)
		self.mainHandle.bind('<Left>', self.left)
		self.mainHandle.bind('<Right>', self.right)
		#self.mainHandle.bind('<a>', self.close)
	
	def left(self, event):
		if self.img_idx != 0:
			self.img_idx -= 1
		self.showIMG(self.img_idx)

	def right(self, event):
		self.img_idx += 1
		self.showIMG(self.img_idx)

	def open_callback(self):
		self.file_path = filedialog.askdirectory(initialdir='.') + '/'
		#self.box_remaining.delete(0,tkinter.END)
		#self.box_remaining.insert(0,file_path)
		
		#print(self.file_path)

	def run(self):
		self.mainHandle.mainloop()
	
		

instance = MainWindow()
instance.run()