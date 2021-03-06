# machine_learning
## Independent Projects in Machine Learning

Here are my projects with a machine learning theme.

### Operating System: 
Windows (unfortunately, all of the filename paths use the Windows convention with "\\")

### Interpreter: 
python2

### Dependencies: 
+ numpy
+ Tkinter
+ gzip
+ cPickle
+ os
+ sys

### List of Projects
1. digit classifier interface
   + To see the demo, run `python nninter.py`. This will load the 3-layer ReLU neural network with default weights, and display a grid, where you can draw a digit and see if the machine recognizes it. 
   + NOTE: As the machine was trained on images centered on the 28x28 grid, try to draw digits that are also centered. The machine will perform better.
2. digit reconstructor
    + To see the demo, run `python dbnRun.py`. This will load the dbn with default weights, train it for 20 epochs, write the new weights to a file and finally display the reconstructed digits.
    + To only see the reconstructed digits using the default weights, run `python dbnDisplay.py`

### Caveats
Please do not change the file structure or rename the files in any way, unless you know what you are doing. The code relies on importing code from vaious places in the file structure, so changing the names or the file structure haphazardly will break the program. :(

### Contact
Please refer any contact to <shguan10@gmail.com> with 'Machine Learning Projects' in the subject header.

### License
© 2017. Xinyu Guan
