import argparse
from numpy.lib.npyio import load as npload
from src.utils import constants


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Subject ID.')
    args = parser.parse_args()

    features_path = constants.data_proc_path+args.subject+'/'+args.subject

    print('Full')
    with open(features_path+'.npy', 'rb') as f:
        x_training = npload(f)
        y_training = npload(f)
        print(x_training.shape)
        print(y_training.shape)

    print('Labeled')
    with open(features_path+'_labeled.npy', 'rb') as f:
        x_training = npload(f)
        y_training = npload(f)
        print(x_training.shape)
        print(y_training.shape)

    print('Notfound')
    with open(features_path+'_notfound.npy', 'rb') as f:
        x_training = npload(f)
        y_training = npload(f)
        print(x_training.shape)
        print(y_training.shape)

    print('Garbage')
    with open(features_path+'_garbage.npy', 'rb') as f:
        x_training = npload(f)
        print(x_training.shape)
