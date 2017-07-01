import matplotlib.pyplot as plt
import time
import threading
import random

class PlotThread():
    def __init__(self):
        self.data = [0]
        self.x = [0]

        thread = threading.Thread(target=self.runCall)
        thread.daemon = True
        thread.start()
        #
        # initialize figure
        plt.figure()
        ln, = plt.plot([])
        plt.ion()
        plt.show()
        while True:
            plt.pause(1)
            ln.set_xdata(self.x)
            ln.set_ydata(self.data)
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
            self.x.append(count)
            print('1')
