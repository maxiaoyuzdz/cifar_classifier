



def main():
    epoch, m, training_loss, validation_loss, training_accuracy, validation_accuracy = ReadAllLossLog(
        '/media/maxiaoyu/data/Log/q11.log')

    shouldv = 20

    va = np.array(validation_accuracy)[::-1]



if __name__ == '__main__':
    main()