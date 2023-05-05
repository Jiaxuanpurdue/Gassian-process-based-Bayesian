import numpy as np
import GPy
import matplotlib.pyplot as plt

from time import time
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
import random as r
import os
import sys

class FileOperation():
    def __init__(self, filename, color,label):
        self.filename = filename
        self.color = color
        self.label=label

    def checkfile(self):
        if os.path.isfile(self.filename)==True:
            print('File already exists.')
            while 1:
                a = input('Which one do you want?\n1. Wipe the file and continue\n2. Append.\n3. Exit. [1/2/3] ')
                if a == '1':
                    with open(self.filename, 'w') as file:
                        pass
                    print('Finish wiping the file.')
                    return
                elif a == '2':
                    return
                elif a == '3':
                    sys.exit("Exit the program now. Please change the file name:)")
                else:
                    print('Wrong input. Please type [1/2/3] again.')
        else:
            with open(self.filename, 'w') as file:
                pass
            print('Finish creating the file.')

    def readdata(self):
        with open(self.filename, 'r') as fo:
            s = fo.read()
        s = s.replace("\n", "")
        s = s[1:len(s) - 1]
        s = s.split('][')
        for i in range(len(s)):
            s[i] = s[i].lstrip()
            s[i] = s[i].rstrip()
            s[i] = s[i].split()
            s_floats = [float(x) for x in s[i]]
            s[i] = s_floats
        # print(s)
        return s

    def plotdata(self, ymin):
        m = np.mean(ymin, axis=0)
        s = np.std(ymin, axis=0)
        # print(m, s)
        plt.errorbar(range(1, len(m) + 1), m, yerr=s, fmt='-o', color=self.color,label=self.label)

        plt.xlabel('number of iterations')
        plt.ylabel('$y_{min}$')
        # plt.show()

    def plotconvergence(self):
        y = self.readdata()
        self.plotdata(y)


txtfile = 'func2_data_opt.txt'
file = FileOperation(filename=txtfile, color='g',label='initial 2')
file.plotconvergence()

txtfile2 = 'func2_opt0505_100.txt'
file2 = FileOperation(filename=txtfile2, color='b',label='initial 1')
file2.plotconvergence()
plt.legend()
plt.show()