import sys
sys.path.insert(0,'..\\rbm')

#Interpreter python2
#trains the dbn for 20 epochs, writes the weights to a file, and displays the reconstructed digits
print("Loading DBN\n")
import cPickle as cP
with open("defValues-300-300.pklb","rb") as f:
	a=cP.load(f)
import dbn
import rbm0_m as rbm0
r=rbm0.rbm0()
r.weights=a[0]
r.load_data()

d=dbn.dbn([r],30)
d.rbmList[1].weights=a[1]

print("Loaded DBN\n")

epochs = 20

print("Training DBN\n")
d.train(epochs)

dweights =[]

print("Saving DBN weights\n")
for x in xrange(len(d.rbmList)):
	dweights.append(d.rbmList[x].weights)

file_write_name="new_weights-784-300-30.pklb"
with open(file_write_name,"wb") as f:
	cP.dump(dweights,f)

print("Displaying DBN reconstructed digits")
import dbninter
dbninter.display(d)