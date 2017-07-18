import sys
import os.path
import argparse
import matplotlib.pyplot as plt
import time
import threading
from logoperator import ReadLossLogByLineNum, ReadAllLossLog
from datautils import str2array

parser = argparse.ArgumentParser(description='Process training arguments')
parser.add_argument('-t', '--work_type', default='s')
parser.add_argument('-e', '--epoch_num', default=300, type=int)
parser.add_argument('-lm', '--loss_max', default=2.0, type=float)
parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/data/Log/')
parser.add_argument('-log', '--log_file_name', default='running.log')
parser.add_argument('-f', '--update_freq', default=1.0, type=float)

parser.add_argument('-ll', '--log_list', type=str2array, nargs='?', default="q1.log",
                    help="input log name")

epoch_data = []
training_loss_data = []
validation_loss_data = []
m_data = []
training_accuracy_data = []
validation_accuracy_data = []

log_line_num = 0

#line declear
training_loss_line = ''
validation_loss_line = ''

training_accuracy_line = ''
validation_accuracy_line = ''

# for compare
compare_epoch_data = []

#compare_training_loss_lines = []
compare_training_loss_data = []
#compare_validation_loss_lines = []
compare_validation_loss_data = []

#compare_training_accuracy_lines = []
compare_training_accuracy_data = []
#compare_validation_accuracy_lines = []
compare_validation_accuracy_data = []



def sleep(seconds):
    for _ in range(0, seconds):
        time.sleep(1)


def checkfileexists(file_path):
    return os.path.isfile(file_path)


def updatedata(log_file_name):
    global log_line_num
    global training_loss_line
    global validation_loss_line

    global training_accuracy_line
    global validation_accuracy_line



    while checkfileexists(log_file_name) is not True:
        print('log file is not exists, wait 5 second')
        sleep(5)

    while True:
        time.sleep(args.update_freq)
        epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy = \
            ReadLossLogByLineNum(log_line_num, log_file_name)
        if epoch == -1:
            pass
        else:
            # training loss
            training_loss_data.append(training_loss)
            # validation loss
            validation_loss_data.append(validation_loss)

            training_accuracy_data.append(training_accuracy)
            validation_accuracy_data.append(validation_accuracy)
            # x
            epoch_data.append(log_line_num)
            log_line_num += 1

            #draw
            training_loss_line.set_xdata(epoch_data)
            training_loss_line.set_ydata(training_loss_data)
            validation_loss_line.set_xdata(epoch_data)
            validation_loss_line.set_ydata(validation_loss_data)

            training_accuracy_line.set_xdata(epoch_data)
            training_accuracy_line.set_ydata(training_accuracy_data)
            validation_accuracy_line.set_xdata(epoch_data)
            validation_accuracy_line.set_ydata(validation_accuracy_data)
            plt.draw()
            plt.pause(0.05)


def showgraphicdynamically(file_name):
    global training_loss_line
    global validation_loss_line
    global training_accuracy_line
    global validation_accuracy_line
    global args

    # draw graphic
    plt.figure()
    plt.subplot(2, 1, 1)
    training_loss_line, = plt.plot([])
    validation_loss_line, = plt.plot([])
    plt.ylim(0, args.loss_max)
    plt.xlim(1, args.epoch_num)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training', 'validation'])

    plt.subplot(2, 1, 2)
    training_accuracy_line, = plt.plot([])
    validation_accuracy_line, = plt.plot([])
    plt.ylim(0, 100)
    plt.xlim(1, args.epoch_num)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('learning accuracy')
    plt.legend(['training', 'validation'])


    plt.ion()
    plt.show()

    thread = threading.Thread(target=updatedata(file_name))
    thread.daemon = True
    thread.start()




def updatestaticdata():
    global log_line_num
    global training_loss_line
    global validation_loss_line

    global epoch_data
    global training_loss_data
    global validation_loss_data
    global m_data
    global training_accuracy_data
    global validation_accuracy_data


    while True:
        time.sleep(0.05)

        training_loss_line.set_xdata(epoch_data)
        training_loss_line.set_ydata(training_loss_data)
        validation_loss_line.set_xdata(epoch_data)
        validation_loss_line.set_ydata(validation_loss_data)

        training_accuracy_line.set_xdata(epoch_data)
        training_accuracy_line.set_ydata(training_accuracy_data)
        validation_accuracy_line.set_xdata(epoch_data)
        validation_accuracy_line.set_ydata(validation_accuracy_data)

        plt.draw()
        plt.pause(0.05)




def showgraphicstaically(file_name):

    if checkfileexists(file_name) is True:

        global training_loss_line
        global validation_loss_line

        global training_accuracy_line
        global validation_accuracy_line

        global args

        global epoch_data
        global training_loss_data
        global validation_loss_data
        global m_data
        global training_accuracy_data
        global validation_accuracy_data

        epoch_data, m_data, training_loss_data, validation_loss_data, training_accuracy_data, validation_accuracy_data = ReadAllLossLog(file_name)


        # draw graphic
        plt.figure()
        plt.subplot(2, 1, 1)
        training_loss_line, = plt.plot([])
        validation_loss_line, = plt.plot([])
        plt.ylim(0, args.loss_max)
        plt.xlim(1, args.epoch_num)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('learning curve')
        plt.legend(['training', 'validation'])

        plt.subplot(2,1,2)
        training_accuracy_line, = plt.plot([])
        validation_accuracy_line, = plt.plot([])
        plt.ylim(0, 100)
        plt.xlim(1, args.epoch_num)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('learning accuracy')
        plt.legend(['training', 'validation'])



        plt.ion()
        plt.show()

        training_loss_line.set_xdata(epoch_data)
        training_loss_line.set_ydata(training_loss_data)
        validation_loss_line.set_xdata(epoch_data)
        validation_loss_line.set_ydata(validation_loss_data)

        training_accuracy_line.set_xdata(epoch_data)
        training_accuracy_line.set_ydata(training_accuracy_data)
        validation_accuracy_line.set_xdata(epoch_data)
        validation_accuracy_line.set_ydata(validation_accuracy_data)

        plt.draw()
        plt.pause(1)

        thread = threading.Thread(target=updatestaticdata())
        thread.daemon = True
        thread.start()

    else:
        print('no file exists')
        pass






def drawCompareGraphic():
    # draw graphic
    plt.figure()

    #draw loss graphic
    plt.subplot(2, 1, 1)

    max_epoch = 0
    max_epoch_data = []
    for i in compare_epoch_data:
        c_max = max(i)
        if c_max > max_epoch:
            max_epoch = c_max
            max_epoch_data = i

    # draw training loss line and compute max training loss value
    max_t_loss = 0
    for t_loss_data in compare_training_loss_data:
        t_loss_line, = plt.plot([])
        #compare_training_loss_lines.append(t_loss_line)
        t_loss_line.set_xdata(max_epoch_data)
        t_loss_line.set_ydata(t_loss_data)
        c_loss = max(t_loss_data)
        if c_loss > max_t_loss:
            max_t_loss = c_loss

    # draw validation loss
    max_v_loss = 0
    for v_loss_data in compare_validation_loss_data:
        v_loss_line, = plt.plot([])
        #compare_validation_loss_lines.append(v_loss_line)
        v_loss_line.set_xdata(max_epoch_data)
        v_loss_line.set_ydata(v_loss_data)
        v_loss = max(v_loss_data)
        if v_loss > max_v_loss:
            max_v_loss = v_loss


    plt.xlim(1, max_epoch)
    plt.ylim(0, max(max_v_loss, max_t_loss))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning curve')

    loss_name_arr = []
    for i in range(0, 2):
        for log_name in args.log_list:
            if i <1:
                t_loss_line_name = log_name.split('.')[0]+'_t_ls'
                loss_name_arr.append(t_loss_line_name)
            else:
                v_loss_line_name = log_name.split('.')[0]+'_v_ls'
                loss_name_arr.append(v_loss_line_name)

    plt.legend(loss_name_arr)

    #draw accuracy graphic
    plt.subplot(2, 1, 2)


    for t_accuracy_data in compare_training_accuracy_data:
        t_accuracy_line, = plt.plot([])
        #compare_training_accuracy_lines.append(t_accuracy_line)
        t_accuracy_line.set_xdata(max_epoch_data)
        t_accuracy_line.set_ydata(t_accuracy_data)

    for v_accuracy_data in compare_validation_accuracy_data:
        v_accuracy_line, = plt.plot([])
        v_accuracy_line.set_xdata(max_epoch_data)
        v_accuracy_line.set_ydata(v_accuracy_data)

    accuracy_name_arr = []
    for i in range(0, 2):
        for log_name in args.log_list:
            if i < 1:
                t_accuracy_line_name = log_name.split('.')[0] + '_t_ac'
                accuracy_name_arr.append(t_accuracy_line_name)
            else:
                v_accuracy_line_name = log_name.split('.')[0] + '_v_ac'
                accuracy_name_arr.append(v_accuracy_line_name)

    plt.xlim(1, max_epoch)
    plt.ylim(0, 100)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy curve')
    plt.legend(accuracy_name_arr)



    #plt.ion()
    plt.show()




    #plt.draw()
    #plt.pause(1)




def showCompareGraphic():
    for log_file in args.log_list:
        if checkfileexists(args.log_dir + log_file):
            epoch_data, m_data, training_loss_data, validation_loss_data, training_accuracy_data, validation_accuracy_data = \
                ReadAllLossLog(args.log_dir + log_file)
            compare_epoch_data.append(epoch_data)
            compare_training_loss_data.append(training_loss_data)
            compare_validation_loss_data.append(validation_loss_data)
            compare_training_accuracy_data.append(training_accuracy_data)
            compare_validation_accuracy_data.append(validation_accuracy_data)
        else:
            print(log_file, ' is not exist')

    if compare_epoch_data.__len__() >= 2:
        drawCompareGraphic()






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

    #draw compare graphic
    if args.work_type == 'c':
        showCompareGraphic()


if __name__ == '__main__':
    main()

