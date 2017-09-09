import tkMessageBox
from Tkinter import *
import numpy as np
import pickle
import network2_ReLU as network
import mnist_loader
import random
import os
from PIL import Image

def checkered(canvas, line_distance,sl):
   # vertical lines at an interval of "line_distance" pixel
   for x in range(line_distance,sl,line_distance):
      canvas.create_line(x, 0, x, sl, fill="#000000")
   # horizontal lines at an interval of "line_distance" pixel
   for y in range(line_distance,sl,line_distance):
      canvas.create_line(0, y, sl, y, fill="#000000")

def average(pixel):
  sum=0
  for x in xrange(len(pixel)):
      sum+=pixel[x]
  return sum/len(pixel)

def load_handwritten_images():
    a = np.zeros((58,784,1))
    y=np.zeros(58)
    i=0
    #import images
    for dirname,dirnames,filenames in os.walk("C:\\Users\\Xinyu\\Documents\\GitHub\\PNG-JPG_MNIST-format\\test-images(mine)"):
        for filename in filenames:
            if filename.endswith('.png'):
                im = Image.open(os.path.join(dirname,filename))
                pix = im.load()

                class_name = int(os.path.join(dirname).split('\\')[-1])
                b=np.zeros((28,28))
                for yy in xrange(28):
                    for xx in xrange(28):
                        try:
                            b[yy][xx]=average(pix[xx,yy])
                        except:
                            b[yy][xx]=pix[xx,yy]#apparently len(int) does not exist
                a[i]=(255-np.reshape(b,(784,1)))/256

                y[i]=class_name#check with nnInter to see if b is actually the image
                i+=1
    return zip(a,y)

class App:
    def __init__(self, master,pl,pn):

        master.title("Recognizing Handwritten Digits")
        frame=Frame(master)
        frame.pack()

        self.master=master

        frame1=Frame(frame)
        frame1.pack(side=TOP)
        frame1_25=Frame(frame)
        frame1_25.pack(side=TOP)
        frame1_5=Frame(frame)
        frame1_5.pack(side=TOP)
        frame2=Frame(frame)
        frame2.pack(side=TOP)

        self.pxlen =pl
        self.pxno=pn
        self.cs=pl*pn

        self.submitButton=Button(frame1,text="Submit Digit", command=self.submit)
        self.submitButton.pack(side=LEFT)

        self.eraseAllB=Button(frame1,text="Start over",command=self.eraseAll)
        self.eraseAllB.pack(side=LEFT)

        self.helpButton=Button(frame1,text="About",command=self.help)
        self.helpButton.pack(side=LEFT)

        self.genTrainImageB=Button(frame1_25,text="Load Random Training Image",command=self.genTrainImage)
        self.genTrainImageB.pack(side=LEFT)
        self.genTestImageB = Button(frame1_25, text="Load Random Test Image", command=self.genTestImage)
        self.genTestImageB.pack(side=LEFT)
        self.genPicImageB=Button(frame1_25,text="Load Random Picture of Handwritten Image", command=self.genPicImage)
        self.genPicImageB.pack(side=LEFT)

        self.train,val,self.test = mnist_loader.load_data_wrapper()
        self.test +=val
        self.handwrittenSet=load_handwritten_images()

        self.v=StringVar()
        r1=Radiobutton(frame1_5,text="Pixel draw",variable=self.v,value="pd",command=self.toggleDraw,indicatoron=0)
        r1.pack(side=LEFT)
        r1_5=Radiobutton(frame1_5,text="Shade draw",variable=self.v,value="sd",command=self.toggleDraw,indicatoron=0)
        r1_5.pack(side=LEFT)
        r2=Radiobutton(frame1_5,text="Pixel erase",variable=self.v,value="pe",command=self.toggleDraw,indicatoron=0)
        r2.pack(side=LEFT)
        r3=Radiobutton(frame1_5,text="Shade erase",variable = self.v,value="se",command=self.toggleDraw,indicatoron=0)
        r3.pack(side=LEFT)
        r1_5.select()
        self.isdrawing = "sd"

        self.canvas = Canvas(frame2, width=self.pxno * self.pxlen, height=self.pxno * self.pxlen)
        self.canvas.pack(anchor=CENTER)

        checkered(self.canvas,pl,self.cs)
        self.drawBox()
        self.canvas.bind("<B1-Motion>",self.toggleBox)
        self.canvas.bind("<Button-1>", self.toggleBox)

        self.nnInput = np.zeros([784,1])
        self.displayData=np.zeros([self.pxno,self.pxno])+255

        self.biases,self.weights=pickle.load(open("C:\\Users\Xinyu\\Documents\\GitHub\\"
                                                  "neural-networks-and-deep-learning_python2\\src\\biasesWeightsReLU784_100_10.pklb","rb"))
        if np.shape(self.biases)[0]!=2: raise ValueError("Number of layers is bigger than 3")
        else: self.net = network.Network([784, int(np.shape(self.biases[0])[0]), 10])#TODO make a button to select which architecture to use
            #TODO make network initialization support more than 3 layers

        self.net.setNet(self.biases,self.weights)# TODO make nn much better. It's optimized on pictures of digits
                                                    # written on paper, not this format. Maybe make a submit digit to
                                                    # training set option so that hopefully the nn can learn what digits
                                                    # digits look like in this format?

    def genPicImage(self):
        self.drawfromset(self.handwrittenSet)

    def genTrainImage(self):
        self.drawfromset(self.train)

    def genTestImage(self):
        self.drawfromset(self.test)

    def drawfromset(self,set):
        self.eraseAll()
        rn = random.random() * np.shape(set)[0]  # TODO implement automatic resizing based on the image found in the mnist_loader
        self.nnInput=set[int(rn)][0]
        self.displayData=np.reshape(255-self.nnInput*256,[self.pxno,self.pxno])
        #data = np.reshape(set[int(rn)][0],(self.pxno,self.pxno))
        for yn in xrange(self.pxno):
            for xn in xrange(self.pxno):
                px = self.displayData[yn][xn]
                (x1, y1, x2, y2) = (xn * self.pxlen, yn * self.pxlen,
                                    (xn + 1) * self.pxlen, (yn + 1) * self.pxlen)
                color = "#" + hex(int(px))[2:] * 3

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        if set!=self.train: print "This image is actually a ", int(set[int(rn)][1])
        else: print "This image is actually a ", np.argmax(set[int(rn)][1])

    def submit(self):
        #print self.net.feedforward(self.nnInput)
        #print np.argmax(self.net.feedforward(self.nnInput))
        n= 255-256*np.reshape(self.nnInput, (self.pxno, self.pxno))
        d=self.displayData
        for y in xrange(self.pxno):
            for x in xrange(self.pxno):
                n[y][x]=int(n[y][x])
                d[y][x]=int(d[y][x])
        if (n!=d).any(): raise ValueError("The display data is not the input")
        a=self.net.feedforward(self.nnInput)
        b = str(np.argmax(a))
        #tkMessageBox.showinfo("Result","I think your digit is a "+ b)
        #print a
        #raise NotImplementedError
        self.submitW=Toplevel(self.master)
        self.submitWapp=submitNNtrain(self.submitW,self.displayData,b)

    def help(self):
        tkMessageBox.showinfo('Recognizing Handwritten Digits',
            'This is an interface to the neural network I trained. Using the draw and erase buttons, draw a digit [0 to 9] to see if '
            'my neural network can recognize it. '
            '\n\n'
            'My neural netowrk was trained using the super-helpful guide found here: \nneuralnetworksanddeeplearning.com ')

    def drawBox(self):
        self.canvas.create_line(0,0,self.cs,0)
        self.canvas.create_line(0, 0, 0, self.cs)
        self.canvas.create_line(self.cs, 0, self.cs, self.cs)
        self.canvas.create_line(0, self.cs, self.cs, self.cs)

    def toggleDraw(self):
        self.isdrawing= self.v.get()

    def eraseAll(self):
        self.canvas.delete(ALL)
        checkered(self.canvas,self.pxlen,self.cs)
        self.drawBox()
        self.nnInput = np.zeros([784, 1])
        self.displayData=np.zeros([self.pxno, self.pxno])+255

    def toggleBox(self, event):
        if event.x>= 0 and event.x<self.cs and event.y>=0 and event.y<self.cs:
            (xn,yn)=(int(event.x/(self.pxlen)),int(event.y/(self.pxlen)))

            state = 0

            if self.isdrawing[1]=="d":
                self.displayData[yn][xn] -= 62
                if self.displayData[yn][xn] < 0: self.displayData[yn][xn] = 0

            if self.isdrawing[0]=="s":
                #raise NotImplementedError
                if self.isdrawing[1]=="d": state=1
                elif self.isdrawing[1]=="e":
                    state=-1
                    self.displayData[yn][xn] += 62
                    if self.displayData[yn][xn] > 255: self.displayData[yn][xn] = 255

                for i in (-1,0,1):
                    for j in (-1,0,1):
                        if((i,j)==(1,1)): pass
                        elif (xn+j)<0 or (xn+j)>=self.pxno: pass
                        elif (yn+i)<0 or (yn+i)>=self.pxno: pass
                        else:
                            self.displayData[yn+i][xn+j] -= state*31
                            if self.displayData[yn+i][xn+j]<0: self.displayData[yn+i][xn+j]=0
                            elif self.displayData[yn+i][xn+j]>255: self.displayData[yn+i][xn+j]=255
                            color = "#" + hex(int(self.displayData[yn+i][xn+j]))[2:] * 3
                            (xj1, yi1, xj2, yi2) = ((xn+j) * self.pxlen, (yn+i)* self.pxlen,
                                                    (xn +j+ 1) * self.pxlen, (yn+i + 1) * self.pxlen)
                            self.canvas.create_rectangle(xj1, yi1, xj2, yi2, fill=color)
            elif self.isdrawing=="pe": self.displayData[yn][xn] = 255

            (cx1,cy1,cx2,cy2)=(xn*self.pxlen,yn*self.pxlen,
                           (xn+1)*self.pxlen,(yn+1)*self.pxlen)
            color = "#"+hex(int(self.displayData[yn][xn]))[2:]*3
            self.canvas.create_rectangle(cx1,cy1,cx2,cy2,fill=color)

            self.nnInput=(255-np.reshape(self.displayData,[784,1]))/256
        else: pass

class submitNNtrain:
    def __init__(self,master,displayData,myClass):
        #top=self.top=Toplevel(parent)
        self.master=master
        Label(master,text="Result").pack()
        Label(master,text="I think your digit is a "+str(myClass)).pack()
        Label(master,text="Is this correct? Enter the correct digit below.").pack()

        self.e = Entry(master)
        self.e.pack(padx=5)

        self.trainImage=displayData

        b=Button(master,text="Submit for training!",command=self.submit)
        b.pack(pady=5)
        b2 = Button(master,text="Cancel",command=self.master.destroy)
        b2.pack(pady=5)

    def submit(self):
        try:
            a = int(self.e.get())
            rn=int(random.random()*1000000)
            im = Image.fromarray(self.trainImage)
            if im.mode != 'RGB':
                im=im.convert('RGB')#TODO implement Learn++
            im.save(".\\customTrain\\"+self.e.get()+"\\"+str(rn)+".png")
            self.master.destroy()
        except ValueError:
            tkMessageBox.showerror("I/O ERROR",
                                   "Please enter the digit the image represents")

root=Tk()
app=App(root,10,28)

root.mainloop()
