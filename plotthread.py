import matplotlib.pyplot as plt
import time
import threading



from logoperator import ReadLog, ReadLogByLineNum



class PlotThread():
    def __init__(self):
        self.data = [0]
        self.data2 = [0]
        self.x = [0]

        self.line_num = 0

        thread = threading.Thread(target=self.runCall)
        thread.daemon = True
        thread.start()
        #
        # initialize figure
        plt.figure()
        ln, = plt.plot([])
        ln2, = plt.plot([])
        plt.ylim(0, 2)
        plt.xlim(1, 300, 1)
        plt.ion()
        plt.show()
        while True:
            plt.pause(1)
            ln.set_xdata(self.x)
            ln.set_ydata(self.data)
            ln2.set_xdata(self.x)
            ln2.set_ydata(self.data2)
            plt.draw()

    def runCall(self):
        time.sleep(1)

        while True:
            #print('line_num = ', self.line_num)
            time.sleep(1)
            epoch, m, training_loss, validation_loss, test_accurate = ReadLogByLineNum(self.line_num)
            if epoch == -1:
                pass
            else:
                #training loss
                self.data.append(training_loss)
                #validation loss
                self.data2.append(validation_loss)
                #x
                self.x.append(self.line_num)
                self.line_num += 1
                #print('show pic')



def main():
    pt = PlotThread()




if __name__ == '__main__':
    main()

