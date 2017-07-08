from pathlib import Path
import fcntl


def SaveLog(epoch, sample_num, training_loss, validation_loss, test_accurate, log_file_name):
    with open(log_file_name, 'a') as the_file:
        fcntl.flock(the_file, fcntl.LOCK_EX)
        the_file.write(str(epoch) +
                       ',' +
                       str(sample_num) +
                       ','
                       + str(training_loss) + ',' +
                       str(validation_loss) + ',' + str(test_accurate) + '\n')


def ReadLossLogByLineNum(line_num, log_file_name):
    with open(log_file_name, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for i, line in enumerate(f):
            if i == line_num:
                if line.__len__() > 0:
                    epoch, m, training_loss, validation_loss, test_accurate = str(line.strip()).split(',')
                    epoch = int(epoch[2:])
                    m = int(m)
                    training_loss = float(training_loss)
                    validation_loss = float(validation_loss)
                    test_accurate = float(test_accurate[:-1])

                    return epoch, m, training_loss, validation_loss, test_accurate

                else:
                    epoch = -1
                    m, training_loss, validation_loss, test_accurate = 0, 0, 0, 0
                    return epoch, m, training_loss, validation_loss, test_accurate

        epoch = -1
        m, training_loss, validation_loss, test_accurate = 0, 0, 0, 0
        return epoch, m, training_loss, validation_loss, test_accurate



def ReadAllLossLog(log_file_name):

    epoch_data = []
    m_data = []
    training_loss_data = []
    validation_loss_data = []
    test_accurate_data = []
    with open(log_file_name, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        for i, line in enumerate(f):
            if line is not 'END':
                epoch, m, training_loss, validation_loss, test_accurate = str(line.strip()).split(',')

                epoch = int(epoch[2:])
                m = int(m)
                training_loss = float(training_loss)
                validation_loss = float(validation_loss)
                test_accurate = float(test_accurate[:-1])

                epoch_data.append(epoch)
                m_data.append(m)
                training_loss_data.append(training_loss)
                validation_loss_data.append(validation_loss)
                test_accurate_data.append(test_accurate)


    return epoch_data, m_data, training_loss_data, validation_loss_data, test_accurate_data

def ReadLogByLineNum(line_num, log_file_name):
    #print('read line num = ', line_num)
    with open('/media/maxiaoyu/datastore/Log/' + log_file_name + '.log', 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for i, line in enumerate(f):
            #print('real i = ', i)

            if i == line_num:
                #print('get line num = ', i)
                if line.__len__() > 0:
                    epoch, m, training_loss, validation_loss, test_accurate = str(line.strip()).split(',')
                    epoch = int(epoch[2:])
                    m = int(m)
                    training_loss = float(training_loss)
                    validation_loss = float(validation_loss)
                    test_accurate = float(test_accurate[:-1])

                    return epoch, m, training_loss, validation_loss, test_accurate

                else:
                    epoch = -1
                    m, training_loss, validation_loss, test_accurate = 0, 0, 0, 0
                    return epoch, m, training_loss, validation_loss, test_accurate

        epoch = -1
        m, training_loss, validation_loss, test_accurate = 0, 0, 0, 0
        return epoch, m, training_loss, validation_loss, test_accurate






def ReadLog(log_file_name):
    with open('/media/maxiaoyu/datastore/Log/' + log_file_name + '.log', 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        #first = f.readline()  # Read the first line.
        for last in f:
            pass
        epoch, m, training_loss, validation_loss, test_accurate = str(last.strip()).split(',')
        print(int(epoch[2:]))
        print(m)
        print(float(training_loss))
        print(float(validation_loss))
        print(float(test_accurate[:-1]))
        epoch = int(epoch[2:])
        m = int(m)
        training_loss = float(training_loss)
        validation_loss = float(validation_loss)
        test_accurate = float(test_accurate[:-1])

        return epoch, m, training_loss, validation_loss, test_accurate








