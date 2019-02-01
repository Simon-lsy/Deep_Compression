import tensorflow as tf
import tensorflow.keras as keras
from copy import deepcopy
import numpy as np
import h5py
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
tf.enable_eager_execution()

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(x_test.shape)

COMPRESSION_RATE = 0.9
BATCH_SIZE = 50
NUM_BATCHES = 1000
NUM_EPOCH = 3
BITS = 5
MAX_SPAN = 2 ** BITS
LEARNING_RATE = 0.001


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=[28, 28, 1]),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

file_name = './result/compressed_model2.h5'


Node = namedtuple('Node', ['frequency', 'value', 'left', 'right'])
Node.__lt__ = lambda x, y: x.frequency < y.frequency


def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


def decode_huffman_tree(code_str):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    idx = 0

    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1':  # Leaf node
            value = bitstr2int(code_str[idx:idx + 32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()


def huffman_decode(data_encoding, codebook_encoding):
    """
    Decodes binary files from directory
    """

    # Read the codebook
    root = decode_huffman_tree(codebook_encoding)

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None:  # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype=int)

weights = dict()

file = h5py.File(file_name, mode='r')
print(file)
layer_weights = dict()
for layer_name in file:
    group = file[layer_name]
    layer_weights[layer_name] = []
    dict_names = [n for n in group]
    if 'index' in dict_names:
        cluster_index = group['cluster_index'][()]
        cluster_index = huffman_decode(cluster_index[0], cluster_index[1])
        index = group['index'][()]
        shape = group['shape'][()]
        centroid = group['centroid'][()]
        weight = np.zeros(shape).flatten()
        pos = int(index[0])
        weight[pos] = centroid[cluster_index[0]]
        i = 1
        position = list()
        while pos < len(weight) and i < len(index):
            pos += index[i]
            position.append(pos)
            if index[i] != MAX_SPAN:
                weight[pos] = centroid[cluster_index[i]]
            i += 1
        weight = weight.reshape(shape)
    else:
        weight = group['weight'][()]
    bias = group['bias'][()]
    weights[layer_name] = [weight, bias]

for layer_id in range(len(model.layers)):
    layer = model.layers[layer_id]
    if layer.name in weights:
        weight = weights[layer.name]
        layer.set_weights(weight)

print('-------------------')
score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])
