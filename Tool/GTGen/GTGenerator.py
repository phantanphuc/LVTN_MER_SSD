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
		self.mainHandle.geometry("750x800")
		

		#####################
		# attribute
		
		self.pending_file_list = []
		self.done_file_list = []
		self.current_folder = ''
		self.current_rect = []
		self.camvas_border = 10
		
		self.dictionary = {}
		self.dictionary_loaded = False
		
		self.writting_line = ''
		self.BB_count = 0
		
		#####################
		
		self.initGUI()
		self.initKeyBinding()
		
		#########################3
		## States
		self.isDragging = False
		self.topleft = (0,0)
		
	def initGUI(self):
	
		############# BROWSE #############################
	
		file_frame = tkinter.Frame(master=self.mainHandle, bd = 10)
		file_frame.pack(fill=tkinter.BOTH)
		
		self.f1Entry = tkinter.Entry(file_frame, bd = 5, width= 50)
		self.f1Entry.pack(side = tkinter.LEFT)
		
		f1Browse = tkinter.Button(file_frame, text = "Browse Folder", width=15, command=self.open_callback)
		f1Browse.pack(side = tkinter.LEFT)
		
		dictBrowse = tkinter.Button(file_frame, text = "Browse Dictionary", width=15, command=self.BrowseDict)
		dictBrowse.pack(side = tkinter.LEFT)
		
		LoadBrowse = tkinter.Button(file_frame, text = "Load", width=15, command=self.loadFile)
		LoadBrowse.pack(side = tkinter.LEFT)

		Debug = tkinter.Button(file_frame, text = "Debug", width=15, command=self.DebugBtn)
		Debug.pack(side = tkinter.LEFT)
		
		##### Choose
		
		file_list_frame = tkinter.Frame(master=self.mainHandle, bd = 10)
		file_list_frame.pack(fill=tkinter.BOTH)
		
		
		label1 = tkinter.Label(file_list_frame, text="Remaining: ", relief=tkinter.RAISED )
		label1.pack(side = tkinter.LEFT)
		

		self.box_remaining = ttk.Combobox(file_list_frame)
		self.box_remaining['values'] = ()
		#self.box.current(0)
		self.box_remaining.pack(side = tkinter.LEFT)
		self.box_remaining.bind("<<ComboboxSelected>>", self.select_file_remaining)
		
		label2 = tkinter.Label(file_list_frame, text="Completed: ", relief=tkinter.RAISED )
		label2.pack(side = tkinter.LEFT)
		

		self.box_completed = ttk.Combobox(file_list_frame)
		self.box_completed['values'] = ()
		self.box_completed.pack(side = tkinter.LEFT)
		
		NextButton = tkinter.Button(file_list_frame, text = "Ok -> Next", width=15, command=self.NextCallback)
		NextButton.pack(side = tkinter.LEFT)
		
		##### IMG
		
		img_frame = tkinter.Frame(master=self.mainHandle, bd = 10)
		img_frame.pack(fill=tkinter.BOTH)
	
		
		#self.img = ImageTk.PhotoImage(Image.open('crop.jpg'))
		self.img = ImageTk.PhotoImage(Image.new('RGB', (512, 256)))
		
		self.img_holder = tkinter.Label(img_frame, image = self.img, borderwidth=10, relief="raised")
		#self.img_holder.pack(side = tkinter.LEFT)

		self.img_canvas = tkinter.Canvas(img_frame, height=256, width=512, borderwidth=self.camvas_border, relief="raised")
		self.img_canvas.pack(side = tkinter.LEFT)
		self.img_canvas.bind("<Button-1>", self.mouseEventDown)
		self.img_canvas.bind("<ButtonRelease-1>", self.mouseEventUp)
		self.img_canvas.bind("<B1-Motion>", self.mouseEventMove)
		# tut: http://effbot.org/tkinterbook/tkinter-events-and-bindings.htm
		self.image_on_canvas = self.img_canvas.create_image(self.camvas_border, self.camvas_border, anchor = tkinter.NW, image = self.img)
		
		
		self.LabelEntry = tkinter.Text(img_frame, bd = 5, width=20, height=1, font=("Helvetica", 16))
		self.LabelEntry.pack(side = tkinter.LEFT)
		
		##################################################
	
		output_holder_frame = tkinter.Frame(master=self.mainHandle, bd = 10)
		output_holder_frame.pack(fill=tkinter.BOTH)
	
		self.output_content = tkinter.Text(output_holder_frame, height=15)
		self.output_content.pack(side = tkinter.LEFT)
	
		#################
		
		save_holder = tkinter.Frame(master=self.mainHandle, bd = 10)
		save_holder.pack(fill=tkinter.BOTH)
		
		saveButton = tkinter.Button(save_holder, text = "Save", width=15, command=self.savefile)
		saveButton.pack(side = tkinter.LEFT)
	


	def DebugBtn(self):
		a = self.output_content.index("end")
		print(self.output_content.index("end"))
		
		
		self.output_content.delete(str(int(a.split('.')[0]) - 1) + '.0', str(int(a.split('.')[0]) + 0) + '.0')
		
		if a.split('.')[0] == '2':
			self.output_content.insert(tkinter.END, 'pokemon\n')
		else:
			self.output_content.insert(tkinter.END, '\npokemon\n')
		
	def mouseEventMove(self, event):
		if self.isDragging:
			if not self.dictionary_loaded:
				return
			
			if len(self.current_rect) != 0:
				self.img_canvas.delete(self.current_rect[-1])
				self.current_rect[-1] = self.img_canvas.create_rectangle(self.topleft[0], self.topleft[1], event.x, event.y, outline='red')
			else:
				pass
			


	def mouseEventDown(self, event):
		if not self.dictionary_loaded:
			tkinter.messagebox.showinfo("Error", "Please browse dictionary first")
			return
		
		if len(self.LabelEntry.get("1.0",tkinter.END)) < 2:
			tkinter.messagebox.showinfo("Error", "Please enter label")
			return
			
		print ("clicked at", event.x, event.y)
		self.isDragging = True
		self.topleft = (event.x, event.y)

		rect = self.img_canvas.create_rectangle(self.topleft[0], self.topleft[1], event.x, event.y, outline='red')
		self.current_rect.append(rect)

	def mouseEventUp(self, event):
		if not self.isDragging:
			return
		self.isDragging = False
		
		tl_x = 0
		tl_y = 0
		
		br_x = 0
		br_y = 0

		w = abs(event.x - self.topleft[0])
		h = abs(event.y - self.topleft[1])
		
		if self.topleft[0] < event.x:
			tl_x = self.topleft[0]
			br_x = event.x
		else:
			br_x = self.topleft[0]
			tl_x = event.x
		
		if self.topleft[1] < event.y:
			tl_y = self.topleft[1]
			br_y = event.y
		else:
			br_y = self.topleft[1]
			tl_y = event.y
		
		label = self.LabelEntry.get("1.0",tkinter.END)
		label_idx = self.dictionary[label.replace('\n', '')]
		
		
		insert_text = str(tl_x) + ' ' + str(tl_y) + ' ' + str(br_x) + ' ' + str(br_y) + ' '
		insert_text = insert_text + str(label_idx) + ' '
		self.output_content.insert(tkinter.INSERT, insert_text)
		
		self.writting_line = self.writting_line + insert_text
		self.BB_count += 1
		
		
	def clearBB(self):
		for rect in self.current_rect:
			self.img_canvas.delete(rect)
		del self.current_rect
		self.current_rect = []
		
		
	def initKeyBinding(self):
		self.mainHandle.bind('<Escape>', self.close)
		#self.mainHandle.bind('<a>', self.close)
	
	def NextCallback(self):
		if not self.dictionary_loaded:
			return
		if len(self.current_rect) == 0:
			return
			
		print(self.writting_line)
		
		spl = self.writting_line.find(' ')
		
		
		fname = self.writting_line[:spl]
		fdata = self.writting_line[spl:][:-1]
		
		self.writting_line = fname + ' ' + str(self.BB_count) + fdata + '\n'
		
		##############################
		idx = self.output_content.index("end")

		self.output_content.delete(str(int(idx.split('.')[0]) - 1) + '.0', str(int(idx.split('.')[0]) + 0) + '.0')
		
		if idx.split('.')[0] == '2':
			self.output_content.insert(tkinter.END, self.writting_line)
		else:
			self.output_content.insert(tkinter.END, '\n' + self.writting_line)
		
		self.clearBB()
		self.updatePendingDone(fname)
	
	def updatePendingDone(self, file):
	
		if file not in self.done_file_list:
	
			self.done_file_list.append(file)
			self.pending_file_list.remove(file)
			
			self.box_remaining['values'] = self.pending_file_list
			self.box_completed['values'] = self.done_file_list
		
		if len(self.pending_file_list) == 0:
			tkinter.messagebox.showinfo("Congrat", "There is no more file!!")
			return
		
		self.box_remaining.current(0)
		self.select_file_remaining()
	
	def open_callback(self):
		file_path = filedialog.askdirectory(initialdir='.')
		#self.box_remaining.delete(0,tkinter.END)
		#self.box_remaining.insert(0,file_path)
		
		for root, dirs, files in os.walk(file_path):
			self.pending_file_list = files
			self.box_remaining['values'] = self.pending_file_list
			self.current_folder = root + '/'
			break

		
	def BrowseDict(self):
		file_path = filedialog.askopenfilename(initialdir='.')
		#self.dictionary
		idx = 0

		with open(file_path) as f:
			content = f.readlines()
			for symbol in content:
				symbol = symbol.replace('\n','')

				split = symbol.split(' ')

				self.dictionary[split[0]] = int(split[1])

		self.dictionary_loaded = True

		
				
	def select_file_remaining(self, event = 0):
	
		if not self.dictionary_loaded:
			tkinter.messagebox.showinfo("Error", "Please browse dictionary first")
			return
	
		self.writting_line = self.box_remaining.get() + ' '
		self.BB_count = 0
	
		#print(self.box_remaining.get())
		self.output_content.insert(tkinter.INSERT, self.box_remaining.get() + ' ')
		self.img = ImageTk.PhotoImage(Image.open(self.current_folder + self.box_remaining.get()))
		#self.img_holder.configure(image=self.img)
		self.img_canvas.itemconfig(self.image_on_canvas, image = self.img)

		
		self.clearBB()
	
	def run(self):
		self.mainHandle.mainloop()
		
	def close(self, event=None):
		self.mainHandle.destroy()

	def savefile(self):
		f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
		if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
			return
		f.write(self.output_content.get('1.0', tkinter.END))
		f.close()
		
	def loadFile(self):
		if len(self.pending_file_list) == 0:
			tkinter.messagebox.showinfo("Error", "Please browse directory first")
			return

		file_path = filedialog.askopenfilename(initialdir='.')

		with open(file_path) as f:
			linecontent = f.readlines()

			for line in linecontent:
				fname = line[:line.find(' ')]
				self.pending_file_list.remove(fname)
				self.done_file_list.append(fname)
				print(fname)

				self.output_content.insert(tkinter.END, line)

		self.box_remaining['values'] = self.pending_file_list
		self.box_completed['values'] = self.done_file_list
		

instance = MainWindow()
instance.run()