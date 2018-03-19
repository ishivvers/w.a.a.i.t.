"""
Library code to build and use a Tiramisu CNN using the Keras functional API
"""

import numpy as np
from keras import layers, regularizers
from keras.models import Model

####################################################
# the model
####################################################

# shared CNN parameters
REGULARIZER_L = 1e-4
DROPOUT_RATE = 0.2
PADDING ='same'
INITIALIZER = 'he_uniform'
POOLING = (2, 2)


def tiramisu(blocks=[4, 5, 7, 10, 12], bottleneck=15,  # architecture of the tiramisu
    n_classes=12, input_shape=(224, 224, 3)):  # properties of the data

    ##########################
    # image input
    _input = layers.Input(shape=input_shape)
    ##########################
    # conv layer
    x = layers.Convolution2D(48, (3, 3), strides=(1, 1),
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(_input)
    ##########################
    # down path
    skips = []
    for nb in blocks:
        x = _dense_block(nb, x, end2end=True)
        skips.append(x)
        x = _transition_down(x)
    ##########################
    # bottleneck
    x = _dense_block(bottleneck, x)
    ##########################
    # up path
    for nb in blocks[::-1]:
        x = layers.concatenate([_transition_up(x), skips.pop()])
        x = _dense_block(nb, x)
    ##########################
    # conv layer
    x = layers.Convolution2D(n_classes, (1, 1), strides=(1, 1),
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    ##########################
    # segmented image output
    x = layers.Activation('softmax')(x)
    _output = layers.Reshape((-1, n_classes))(x)
    ####################################################
    # put it together
    model = Model(inputs=_input, outputs=_output)
    return model


def _layer(x):
    filters = 16  # the growth rate (# of feature maps added per layer)
    kernel = (3, 3)
    stride = (1, 1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(filters, kernel, strides=stride,
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return x


def _dense_block(n_layers, x, end2end=False):
    # if end2end, will provide a path from input to output, as well as internal paths
    _these_layer_outputs = []
    _start = x
    # n-1 layers with their outputs concatted to their inputs
    for i in range(n_layers-1):
        lyr = _layer(x)
        _these_layer_outputs.append(lyr)
        x = layers.concatenate([x, lyr])
    # one more layer, then concatenate all of the layer outputs
    _these_layer_outputs.append(_layer(x))
    if end2end:
        _these_layer_outputs.append(_start)
    x = layers.concatenate(_these_layer_outputs)
    return x


def _transition_down(x):
    filters = x.get_shape().as_list()[-1]  # same number of output feature maps as input
    kernel = (1, 1)
    stride = (1, 1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(filters, kernel, strides=stride,
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.MaxPooling2D(pool_size=POOLING, strides=POOLING)(x)
    return x


def _transition_up(x):
    filters = x.get_shape().as_list()[-1]  # same number of output feature maps as input
    kernel = (3, 3)
    x = layers.Conv2DTranspose(filters, kernel, strides=POOLING,
                               padding=PADDING, kernel_initializer=INITIALIZER,
                               kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    return x


####################################################
# data generator
####################################################
class SegmentGenerator(object):
    """
    Given a data set (X) and associated labels (Y), a batchsize and
    the expected image size, generate batches of image data of out_sz by
    randomly cropping the (assumed to be larger) input images.
    If train==True, augments the data by randomly
    flipping the images horizontally and vertically.
    """
    def __init__(self, x, y, batchsize=5, out_sz=(224, 224), train=False):
        self.x = x
        self.y = y
        self.batchsize = batchsize
        self.train = train
        self.n_image, self.ri, self.ci, _ = x.shape
        self.ro, self.co = out_sz
        # make the labels tensor 4D
        self.y = self.y[:, :, :, np.newaxis]

    def get_slice(self, i,o):
        start = np.random.randint(0, i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        """
        Produce a sliced/cropped subimage from the image with index idx.
        If training, augment via random rotations and flips.
        """
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (np.random.random() > 0.5):
            # flip vertically
            y = y[::-1, :, :]
            x = x[::-1, :, :]
        if self.train and (np.random.random() > 0.5):
            # flip horizontally
            y = y[:, ::-1, :]
            x = x[:, ::-1, :]
        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        idxs = np.random.choice(range(self.n_image), self.batchsize)
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        xs = np.stack(xs)
        ys = np.stack(ys).reshape((self.batchsize, -1, 1))
        return xs, ys

    def next(self):
        # make it python 2 & 3 compatible
        return self.__next__()


class SegmentGeneratorIndexed(object):
    """
    Given a data set (X), a batchsize and the expected image size,
    generate batches of image data of out_sz by
    cropping the (assumed to be larger) input images.
    Returns image subarrays and their associated start indices within the input
    image.
    """
    def __init__(self, x, batchsize=5, out_sz=(224, 224)):
        self.x = x
        self.batchsize = batchsize
        self.shape = x.shape
        self.ro, self.co = out_sz
        self.start_coords = []
        for i in range(self.shape[0] // self.ro):
            for j in range(self.shape[1] // self.co):
                self.start_coords.append((i*self.ro, j*self.co))
        self.idx = 0

    def get_item(self, idx):
        """
        Produce a sliced/cropped subimage.
        """
        r, c = self.start_coords[idx]
        return self.x[r: r + self.ro, c: c + self.co], (r, c)

    def __iter__(self):
        return self

    def __next__(self):
        xs = np.zeros((self.batchsize, self.ro, self.co, self.shape[2]))
        coords = []
        for i in range(self.batchsize):
            xs[i], c = self.get_item(self.idx)
            coords.append(c)
            self.idx += 1
            if self.idx >= len(self.start_coords):
                raise StopIteration
        return xs, coords

    def next(self):
        # make it python 2 & 3 compatible
        return self.__next__()
