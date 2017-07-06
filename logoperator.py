from pathlib import Path
import fcntl


def SaveLog(epoch, sample_num, training_loss, validation_loss, test_accurate, log_file_name):
    with open('/media/maxiaoyu/datastore/Log/' + log_file_name + '.log', 'a') as the_file:
        fcntl.flock(the_file, fcntl.LOCK_EX)
        the_file.write(str(epoch) +
                       ',' +
                       str(sample_num) +
                       ','
                       + str(training_loss) + ',' +
                       str(validation_loss) + ',' + str(test_accurate) + '\n')




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
"""
        fcntl.flock(f, fcntl.LOCK_EX)
        first = f.readline()  # Read the first line.
        f.seek(-2, 2)  # Jump to the second last byte.
        while f.read(1) != b"\n":  # Until EOL is found...
            f.seek(-2, 1)  # ...jump back the read byte plus one more.
        last = f.readline().strip()
        print(last)
"""



"""
        first = next(fh)
        offs = -100
        while True:
            fh.seek(offs, 2)
            lines = fh.readlines()
            if len(lines) > 1:
                last = lines[-1]
                break
            offs *= 2
        print(first)
        print(last)



        
        content = f.readline().strip()
        print(content)
        print(content.__len__())
        epoch, m, training_loss, validation_loss, test_accurate = content.split(',')
        print(test_accurate)
        print('==')
        content = f.readline()
        print(content)
        print(content.__len__())
        print('==')
        content = f.readline()
        print(content)
        print(content.__len__())
        print('==')
        #lines = [line.rstrip('\n') for line in f]
        #print(lines)
        """







