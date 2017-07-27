

import numpy as np
from logoperator import ReadAllLossLog




def main():
    epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy = ReadAllLossLog('/media/maxiaoyu/data/Log/q11.log')

    shouldv = 20


    va = np.array(validation_accuracy)[::-1]

    max_va = va.max()
    # print(max_va)
    va_std = []
    # print(va.size)
    for index in np.arange(0, shouldv, 5):
        # print(va[index: index + 5], va[index: index + 5].std())
        va_std.append(va[index: index + 5].std())

    # key 1
    va_std_avg = np.mean(va_std)
    # print(va_std_avg)
    va_max_small = va[0:shouldv].max()
    # print(va_max_small)
    # key 2
    max_dis = np.abs(max_va - va_max_small)
    # print(max_dis)
    if va_std_avg <= 0.07 and max_dis <= 0.2:
        print('end')
        #return True
    else:
        print('not end')
        #return False





    print('end')



if __name__ == '__main__':
    main()