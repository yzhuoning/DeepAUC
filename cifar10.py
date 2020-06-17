import tarfile
from six.moves import urllib
import sys
import numpy as np
import pickle
import os

data_dir = 'cifar10_data'
full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE  = 10000 * NUM_TRAIN_BATCH

def maybe_download_and_extract():
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path):
    fo = open(path, 'rb')
    dicts = pickle.load(fo,  encoding='latin1')
    fo.close()

    data = dicts['data']
    label = np.array(dicts['labels'])
  
    return data, label


def read_in_all_images(address_list, shuffle=True):
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        #print ('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        #print ('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label


def read_train_data(padding_size=2):
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list)
    return data, label

def read_test_data():
    validation_array, validation_labels = read_in_all_images([vali_dir])
    return validation_array, validation_labels

def load_data():
    all_data, all_labels  = read_train_data()
    test_data, test_labels = read_test_data()
    return (all_data, all_labels), (test_data, test_labels)
   
    
maybe_download_and_extract()