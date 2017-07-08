import sys
import os.path
import matplotlib.pyplot as plt
import time
import threading
from logoperator import ReadLossLogByLineNum, ReadAllLossLog

WORK_TYPE = 's'
LOG_FILE_NAME = ''
EPOCH_NUM = 300
LOSS_MAX = 2

epoch_data = []
training_loss_data = []
validation_loss_data = []
m_data = []
test_accurate_data = []

log_line_num = 0

#line declear
training_line = ''
validation_line = ''


def sleep(seconds):
    for _ in range(0, seconds):
        time.sleep(1)


def checkfileexists(file_path):
    return os.path.isfile(file_path)


def updatedata(log_file_name):
    global log_line_num
    global training_line
    global validation_line

    while checkfileexists(log_file_name) is not True:
        print('log file is not exists, wait 5 second')
        sleep(5)

    while True:
        time.sleep(1)
        epoch, m, training_loss, validation_loss, test_accurate = ReadLossLogByLineNum(log_line_num, log_file_name)
        if epoch == -1:
            pass
        else:
            # training loss
            training_loss_data.append(training_loss)
            # validation loss
            validation_loss_data.append(validation_loss)
            # x
            epoch_data.append(log_line_num)
            log_line_num += 1

            #draw
            training_line.set_xdata(epoch_data)
            training_line.set_ydata(training_loss_data)
            validation_line.set_xdata(epoch_data)
            validation_line.set_ydata(validation_loss_data)
            plt.draw()
            plt.pause(1)







def showgraphicdynamically(file_name):
    global training_line
    global validation_line
    global LOSS_MAX
    global EPOCH_NUM

    # draw graphic
    plt.figure()
    training_line, = plt.plot([])
    validation_line, = plt.plot([])
    plt.ylim(0, LOSS_MAX)
    plt.xlim(1, EPOCH_NUM)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training', 'validation'])
    plt.ion()
    plt.show()

    thread = threading.Thread(target=updatedata(file_name))
    thread.daemon = True
    thread.start()




def updatestaticdata():
    global log_line_num
    global training_line
    global validation_line

    global epoch_data
    global training_loss_data
    global validation_loss_data
    global m_data
    global test_accurate_data


    while True:
        time.sleep(0.05)
        #print('drawing...')
        training_line.set_xdata(epoch_data)
        training_line.set_ydata(training_loss_data)
        validation_line.set_xdata(epoch_data)
        validation_line.set_ydata(validation_loss_data)
        plt.draw()
        plt.pause(0.05)




def showgraphicstaically(file_name):

    if checkfileexists(file_name) is True:

        global training_line
        global validation_line
        global LOSS_MAX
        global EPOCH_NUM

        global epoch_data
        global training_loss_data
        global validation_loss_data
        global m_data
        global test_accurate_data

        epoch_data, m_data, training_loss_data, validation_loss_data, test_accurate_data = ReadAllLossLog(file_name)


        # draw graphic
        plt.figure()
        training_line, = plt.plot([])
        validation_line, = plt.plot([])
        plt.ylim(0, LOSS_MAX)
        plt.xlim(1, EPOCH_NUM)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('learning curve')
        plt.legend(['training', 'validation'])
        plt.ion()
        plt.show()

        training_line.set_xdata(epoch_data)
        training_line.set_ydata(training_loss_data)
        validation_line.set_xdata(epoch_data)
        validation_line.set_ydata(validation_loss_data)
        plt.draw()
        plt.pause(1)


        thread = threading.Thread(target=updatestaticdata())
        thread.daemon = True
        thread.start()

    else:
        print('no file exists')
        pass





def main():
    print('draw graphic')
    # do drmatically drawing
    if WORK_TYPE == 's':
        showgraphicstaically(LOG_FILE_NAME)
        while True:
            time.sleep(1)

    #draw static graphic
    if WORK_TYPE == 'd':
        showgraphicdynamically(LOG_FILE_NAME)



if __name__ == '__main__':
    WORK_TYPE = sys.argv[1]
    LOG_FILE_NAME = sys.argv[2]

    if str(LOG_FILE_NAME).split('/').__len__() == 1:
        LOG_FILE_NAME = '/media/maxiaoyu/datastore/Log/' + str(LOG_FILE_NAME)

    if sys.argv.__len__() > 3:
        EPOCH_NUM = int(sys.argv[3])
    if sys.argv.__len__() > 4:
        LOSS_MAX = int(sys.argv[4])
    main()

