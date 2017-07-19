import sys
import argparse

count = 99910

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2array(s):
    return s.split(',')

parser = argparse.ArgumentParser(description='Process training arguments')
parser.add_argument('-e', '--epoch', default=300, type=int)
parser.add_argument('-mb', '--mini_batch_size', default=10, type=int)
parser.add_argument('-tb', '--test_batch_size', default=4, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/datastore/Log/')
parser.add_argument('-log', '--log_file_name', default='running.log')
parser.add_argument('-r', '--resume', type=str2bool, nargs='?',
                    const=True, default="True",
                    help="Activate nice mode.")

parser.add_argument('-l', '--log_list', type=str2array, nargs='?', default="q1.log",
                    help="input log name")



def main():
    print('ok')
    args = parser.parse_args()
    if args.resume:
        print('resume')
    else:
        print('no resume')

    print(args)

    input('Press Enter To Continue')


if __name__ == '__main__':
    main()
