import numpy as np
import Tkinter as tk

#set is assumed to be a matrix where each column vector is a flattened 28x28 image with values~[-255,255]
#each image vector must be true color (not inverted, etc)
def display(set):
    if len(np.shape(set))!=2 or np.shape(set)[0]!=784 or (set <-255).any() or (set>255).any():
        print "shape of set", np.shape(set)
        print "set< -255 ", (set <-255).any()
        print "set> 255 ", (set>255).any()
        raise ValueError("Set must be a matrix where each column vector is a flattened 28x28 image with values~[-255,255]\neach image vector must not be inverted")
    root = tk.Tk()
    app = App(root, set)
    root.mainloop()

class App:
    # set is assumed to be a matrix where each column vector is a flattened 28x28 image with values~[0,255]
    def __init__(self,master,set):
        self.master = master
        self.set=set
        master.title("Image Vector Displayer")
        frame=tk.Frame(master)
        frame.pack()

        frame1=tk.Frame(frame)
        frame1.pack(side=tk.TOP)
        frame2=tk.Frame(frame)
        frame2.pack(side=tk.TOP)

        self.prevButton = tk.Button(frame1, text="Previous", command=self.prev)
        self.prevButton.pack(side=tk.LEFT)
        self.nextButton=tk.Button(frame1,text="Next",command=self.nxt)
        self.nextButton.pack(side=tk.LEFT)

        self.pxlen=20
        self.cs=self.pxlen*28
        self.canvas=tk.Canvas(frame2,width=self.cs,height=self.cs)
        self.canvas.pack(anchor=tk.CENTER)

        self.currentImage=0
        self.draw(self.set[:,self.currentImage])
        print "index: ", self.currentImage

    def prev(self):
        self.currentImage-=1;
        self.draw(self.set[:,self.currentImage])
        print "index: ",self.currentImage

    def nxt(self):
        self.currentImage += 1;
        self.draw(self.set[:, self.currentImage])
        print "index: ", self.currentImage

    #np.shape(image) should be (784,)
    #image must have values ~ [0,255]
    def draw(self,image):
        self.eraseAll()
        imArray=np.reshape(image,(28,28))
        #imArray=imArray
        for yn in xrange(28):
            for xn in xrange(28):
                px = imArray[yn][xn]
                (x1, y1, x2, y2) = (xn * self.pxlen, yn * self.pxlen,
                                    (xn + 1) * self.pxlen, (yn + 1) * self.pxlen)
                color = ""
                if px>=0:
                    p=hex(int(255-px))[2:]
                    if len(p)!=2: p="0"+p
                    color = "#FF" + p * 2 #pos is red
                else:
                    p = hex(int(255 + px))[2:]
                    if len(p) != 2: p = "0" + p
                    color = "#" + p * 2 + "FF" #neg is blue
                    #if px<-25: print "px less than -25: ", px
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    #draws the box around the checkered canvas
    def drawBox(self):
        self.canvas.create_line(0,0,self.cs,0)
        self.canvas.create_line(0, 0, 0, self.cs)
        self.canvas.create_line(self.cs, 0, self.cs, self.cs)
        self.canvas.create_line(0, self.cs, self.cs, self.cs)

    def checkered(self):
        # vertical lines at an interval of "line_distance" pixel
        for x in range(self.pxlen, self.cs, self.pxlen):
            self.canvas.create_line(x, 0, x, self.cs, fill="#000000")
        # horizontal lines at an interval of "line_distance" pixel
        for y in range(self.pxlen, self.cs, self.pxlen):
            self.canvas.create_line(0, y, self.cs, y, fill="#000000")

    #erases everything on the display
    def eraseAll(self):
        self.canvas.delete(tk.ALL)
        self.checkered()
        self.drawBox()
