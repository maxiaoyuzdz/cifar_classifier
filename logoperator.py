from pathlib import Path
import fcntl


def SaveLog(epoch, sample_num, training_loss, validation_loss, training_accuracy, validation_accuracy, log_file_name):
    with open(log_file_name, 'a') as the_file:
        fcntl.flock(the_file, fcntl.LOCK_EX)
        the_file.write(
            '{0},{1},{2},{3},{4},{5}\n'.format(str(epoch), str(sample_num), str(training_loss), str(validation_loss),
                                               str(training_accuracy), str(validation_accuracy)))


def ReadLossLogByLineNum(line_num, log_file_name):
    with open(log_file_name, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for i, line in enumerate(f):
            if i == line_num:
                if line.__len__() > 0:
                    epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy = str(line.strip()).split(',')
                    epoch = int(epoch[2:])
                    m = int(m)
                    training_loss = float(training_loss)
                    validation_loss = float(validation_loss)
                    training_accuracy = float(training_accuracy)
                    validation_accuracy = float(validation_accuracy[:-1])

                    return epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy

                else:
                    epoch = -1
                    m, training_loss, validation_loss, training_accuracy, validation_accuracy = 0, 0, 0, 0, 0
                    return epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy

        epoch = -1
        m, training_loss, validation_loss, training_accuracy, validation_accuracy = 0, 0, 0, 0, 0
        return epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy


def ReadAllLossLog(log_file_name):

    epoch_data = []
    m_data = []
    training_loss_data = []
    validation_loss_data = []
    training_accuracy_data = []
    validation_accuracy_data = []
    with open(log_file_name, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        for i, line in enumerate(f):
            if line is not 'END':
                epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy = str(line.strip()).split(',')

                epoch = int(epoch[2:])
                m = int(m)
                training_loss = float(training_loss)
                validation_loss = float(validation_loss)
                training_accuracy = float(training_accuracy)
                validation_accuracy = float(validation_accuracy[:-1])

                epoch_data.append(epoch)
                m_data.append(m)
                training_loss_data.append(training_loss)
                validation_loss_data.append(validation_loss)
                training_accuracy_data.append(training_accuracy)
                validation_accuracy_data.append(validation_accuracy)

    return epoch_data, m_data, training_loss_data, validation_loss_data, training_accuracy_data, validation_accuracy_data







