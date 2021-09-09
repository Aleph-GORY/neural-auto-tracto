import numpy as np
import argparse

from numpy.lib.npyio import load

training_data_path = '../data/training/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Subject ID.')
    # parser.add_argument('--npoints', default='20', help='Number of points in each streamline. (default: 20)')
    args = parser.parse_args()

    load_dir = training_data_path+args.subject+'/'

    print('Full')
    with open(load_dir+args.subject+'.npy', 'rb') as f:
        x_training = np.load(f)
        y_training = np.load(f)
        print(x_training.shape)
        print(y_training.shape)

    print('Labeled')
    with open(load_dir+args.subject+'_labeled.npy', 'rb') as f:
        x_training = np.load(f)
        y_training = np.load(f)
        print(x_training.shape)
        print(y_training.shape)

    print('Notfound')
    with open(load_dir+args.subject+'_notfound.npy', 'rb') as f:
        x_training = np.load(f)
        y_training = np.load(f)
        print(x_training.shape)
        print(y_training.shape)

    print('Garbage')
    with open(load_dir+args.subject+'_garbage.npy', 'rb') as f:
        x_training = np.load(f)
        print(x_training.shape)