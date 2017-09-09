import numpy as np
import Tkinter as tk
import rbm0_m as rbm0

#set is assumed to be a matrix where each column vector is a flattened 28x28 image
#will linearly scale the display image to 0,255 scale
#each image vector must be true color (not inverted, etc)
def display(rbm):
    if rbm.datasorted:
        set = rbm.data[5][:-1]
    else:
        set = rbm.data[:-1]
    #set = rbm.weights.T[:-1]
    if len(np.shape(set))!=2 or np.shape(set)[0]!=784:
        print "shape of set", np.shape(set)
        print "set< -255 ", (set <-255).any()
        print "set> 255 ", (set>255).any()
        raise ValueError("Set must be a matrix where each column vector is a flattened 28x28 image with values~[-255,255]\neach image vector must not be inverted")
    root = tk.Tk()
    app = App(root, set,rbm)
    root.mainloop()

def defdisplay():
    r=rbm0.rbm0()
    r.load_binary_images()
    import cPickle as cP
    r.weights=cP.load(open("rbmWeights_300h.pklb","rb"))
    display(r)

class App:
    # set is assumed to be a matrix where each column vector is a flattened 28x28 image with values~[0,255]
    def __init__(self,master,set,rbm):
        self.master = master
        self.set=set
        self.rbm=rbm
        master.title("RBM interface")
        frame=tk.Frame(master)
        frame.pack()

        frame1=tk.Frame(frame)
        frame1.pack(side=tk.TOP)
        frame1_5=tk.Frame(frame)
        frame1_5.pack(side=tk.TOP)
        frame2=tk.Frame(frame)
        frame2.pack(side=tk.TOP)

        self.prevButton = tk.Button(frame1, text="Previous", command=self.prev)
        self.prevButton.pack(side=tk.LEFT)
        self.nextButton=tk.Button(frame1,text="Next",command=self.nxt)
        self.nextButton.pack(side=tk.LEFT)

        self.convertButton=tk.Button(frame1_5,text="Reconstruct through RBM",command=self.reconstruct)
        #self.convertButton = tk.Button(frame1_5, text="Reconstruct through RBM", command=self.displayreconstruct)
        self.convertButton.pack(side=tk.LEFT)

        self.pxlen=20
        self.cs=self.pxlen*28
        self.canvas=tk.Canvas(frame2,width=self.cs,height=self.cs)
        self.canvas.pack(anchor=tk.CENTER)

        self.currentImage=0
        self.draw(self.set[:,self.currentImage]*255)
        print "index: ", self.currentImage

    def displayreconstruct(self):
        wsinb = self.rbm.weights[:, :-1]
        image=self.set[:,self.currentImage]
        image= np.reshape(image, (784, 1))
        bias=self.rbm.weights[:,-1]
        bh= np.reshape(bias,(len(bias),1))
        zsinb=np.dot(wsinb,image)

        h=rbm0.sigmoid(zsinb+bh)
        h[:-1]=1
        #the bug here is that we use the probabilities, not a sample

        v=rbm0.sigmoid(np.dot(self.rbm.weights.T,h))
        #print np.shape(v)
        self.draw(v[:-1])#*255/np.amax(v[:-1]))

    def reconstruct(self):
        # set the visible units to be the training vector
        # visible = np.reshape(np.append(train_vs,1),(785,1))
        #print "mini_batch avg: ", np.mean(train_vs)
        # update hidden
        train_vs = np.reshape(np.append(self.set[:, self.currentImage], [1], axis=0), (785, 1))
        #print self.rbm.weights
        hiddenprobs = rbm0.sigmoid(np.dot(self.rbm.weights, train_vs))
        hiddenprobs[-1] = 1  # fix the 1s for the hidden biases
        hidden = hiddenprobs > np.random.rand(self.rbm.size[1] + 1, len(train_vs[0]))

        # reconstruct visible, but only use probabilities
        reconVprob = rbm0.sigmoid(np.dot(self.rbm.weights.T, hidden))
        #recon = reconVprob > np.random.rand(len(reconVprob), len(reconVprob[0]))
        # return weight changes
        #self.draw(reconVprob[:-1]*255)
        self.draw(reconVprob[:-1]*255)

    def prev(self):
        self.currentImage-=1;
        self.draw(self.set[:,self.currentImage]*255)#*255/np.amax(np.absolute(self.set[:,self.currentImage])))
        print "index: ",self.currentImage

    def nxt(self):
        self.currentImage += 1;
        self.draw(self.set[:, self.currentImage]*255)#*255/np.amax(np.absolute(self.set[:,self.currentImage])))
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
                    if len(p)<2: p="0"+p
                    color = "#FF" + p * 2 #pos is red
                else:
                    print int(255+px)
                    p = hex(int(255 + px))[2:]
                    if len(p) <2: p = "0" + p
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
