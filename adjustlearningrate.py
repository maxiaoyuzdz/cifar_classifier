
import numpy as np
from logoperator import ReadAllLossLog


def main():
    epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy = ReadAllLossLog(
        '/media/maxiaoyu/data/Log/q11.log')

    shouldv = 400

    va = np.array(validation_accuracy)
    max_va = va.max()
    print(max_va)
    va_std = []
    va_mean = []
    print(va.size)
    for index in np.arange(0, shouldv, 5):
        vt = va[index: index + 5]
        print(index, index + 5, vt, vt.max(), vt.min(), (vt.max() - vt.min()) / 5.0, vt.mean(), vt.std())
        va_std.append(vt.std())
        va_mean.append(vt.mean())

    for mindex in np.arange(0, va_mean.__len__(), 2):
        print(mindex, mindex+1, va_mean[mindex:mindex + 2], va_std[mindex:mindex + 2])

    # key 1
    va_std_avg = np.mean(va_std)
    print(va_std_avg)
    va_max_small = va[0:shouldv].max()
    print(va_max_small)
    # key 2
    max_dis = np.abs(max_va - va_max_small)
    print(max_dis)

    if va_std_avg <= 0.07 and max_dis <= 0.2:
        print('end')
    else:
        print('not end')


if __name__ == '__main__':
    main()