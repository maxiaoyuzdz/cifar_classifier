import sys
import os.path
import argparse
import matplotlib.pyplot as plt
import time
import threading
from logoperator import ReadLossLogByLineNum, ReadAllLossLog

parser = argparse.ArgumentParser(description='Process training arguments')
parser.add_argument('-t', '--work_type', default='s')
parser.add_argument('-e', '--epoch_num', default=300, type=int)
parser.add_argument('-lm', '--loss_max', default=2.0, type=float)
parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/data/Log/')
parser.add_argument('-log', '--log_file_name', default='running.log')

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
    global args

    # draw graphic
    plt.figure()
    training_line, = plt.plot([])
    validation_line, = plt.plot([])
    plt.ylim(0, args.loss_max)
    plt.xlim(1, args.epoch_num)
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

        global args

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
        plt.ylim(0, args.loss_max)
        plt.xlim(1, args.epoch_num)
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
    global args
    args = parser.parse_args()


    print('draw graphic')
    # do drmatically drawing
    if args.work_type == 's':
        showgraphicstaically(args.log_dir + args.log_file_name)
        while True:
            time.sleep(1)

    #draw static graphic
    if args.work_type == 'd':
        showgraphicdynamically(args.log_dir + args.log_file_name)


if __name__ == '__main__':
    main()

