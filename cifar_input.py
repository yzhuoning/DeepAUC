import numpy as np
import cifar10

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3

def per_image_standardization(image_np):
    '''
    Ref: https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np

def random_flip_left_right(image, axis):
    '''
    Ref: https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = np.flip(image, axis)
    return image

def random_crop_and_flip(batch_data, padding_size=2):
    '''
    Ref: https://www.tensorflow.org/api_docs/python/tf/image/random_crop
    '''    
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT, y_offset:y_offset+IMG_WIDTH, :]
        cropped_batch[i, ...] = random_flip_left_right(image=cropped_batch[i, ...], axis=1)
    return cropped_batch

def padding(data, padding_size=2):
    '''
    Ref: https://www.tensorflow.org/api_docs/python/tf/image/random_crop
    '''
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    return data


def load_data(dataset=10, is_tune=False, is_crop_filp=False):
    
    if dataset == 10:
        (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
        
    if is_tune:
        test_data = train_data[-10000:]
        test_labels = train_labels[-10000:]
        train_data = train_data[:-10000]
        train_labels = train_labels[:-10000]
    
    #  (N, 1) --> (N,)    
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)  
    
    # per image standarizartion
    test_data = per_image_standardization(test_data)
    
    # use the below line only when online setting
    if is_crop_filp:
        train_data = padding(train_data)
        train_data = random_crop_and_flip(train_data, padding_size=2)
    else:
        train_data = train_data
    train_data = per_image_standardization(train_data)
    
    print ('Loading dataset: [CIFAR%d], is_tune: [%s], is_preprocessed: [%s], is_crop_filp:[%s]'%(dataset, is_tune, 'True', str(is_crop_filp)))
    print ('Train_data: {}, Test_data: {}'.format(train_data.shape, test_data.shape))
    return (train_data, train_labels), (test_data, test_labels)


def generate_augment_train_batch(train_data, train_labels, train_batch_size, is_tune=False):
        '''
        Ref: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
        '''
        EPOCH_SIZE = 50000 if not is_tune else 50000-5000
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = padding(batch_data)
        batch_data = random_crop_and_flip(batch_data, padding_size=2)
        #batch_data = per_image_standardization(batch_data)  
        batch_label = train_labels[offset:offset+train_batch_size]
        return batch_data, batch_label
    
