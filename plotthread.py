import matplotlib.pyplot as plt
import time
import threading
import random

from pathlib import Path
from logoperator import ReadLog


q = Path('/media/maxiaoyu/datastore/somefile.txt')



class PlotThread():
    def __init__(self):
        self.data = [0]
        self.data2 = [0]
        self.x = [0]

        thread = threading.Thread(target=self.runCall)
        thread.daemon = True
        thread.start()
        #
        # initialize figure
        plt.figure()
        ln, = plt.plot([])
        ln2, = plt.plot([])
        plt.ylim(0, 5)
        plt.xlim(0, 10, 1)
        plt.ion()
        plt.show()
        while True:
            plt.pause(1)
            ln.set_xdata(self.x)
            ln.set_ydata(self.data)
            ln2.set_xdata(self.x)
            ln2.set_ydata(self.data2)
            plt.draw()
            print('3')


    def runCall(self):
        count = 0
        print('22')
        time.sleep(1)
        while True:
            count += 1
            time.sleep(1)
            self.data.append(random.random())
            self.data2.append(random.random() * 2)
            self.x.append(count)
            print('1')




pt = PlotThread()