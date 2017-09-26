import sys
sys.path.insert(0,'..\\rbm')

#Interpreter python2
#displays the dbn reconstructed digits
print("Loading DBN\n")
import cPickle as cP
with open("..\\..\\data\\defValues-300-300.pklb","rb") as f:
	a=cP.load(f)
import dbn
import rbm0_m as rbm0
r=rbm0.rbm0()
r.weights=a[0]
r.load_data()
d=dbn.dbn([r],30)
d.rbmList[1].weights=a[1]

print("Displaying DBN\n")
import dbninter
dbninter.display(d)