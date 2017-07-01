import matplotlib.pyplot as plt
import time
import threading
import random

data = [0]
x = [0]

# This just simulates reading from a socket.
def data_listener():
    count = 0
    while True:
        count += 1
        time.sleep(1)
        data.append(random.random())
        x.append(count)
        print('1')


if __name__ == '__main__':
    thread = threading.Thread(target=data_listener)
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
        ln.set_xdata(x)
        ln.set_ydata(data)
        plt.draw()


    print('nice')