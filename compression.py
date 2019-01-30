import tensorflow as tf
from sklearn.cluster import KMeans
import tensorflow.keras as keras
from copy import deepcopy
import numpy as np
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


# history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=50)
# score = model.evaluate(x_test, y_test)
# print(score[1])

# model.save_weights('./result/my_model.h5', save_format='h5')

model.load_weights('./result/my_model.h5')
score = model.evaluate(x_test, y_test)
print(score[1])


def get_batch(batch_size):
    index = np.random.randint(0, np.shape(x_train)[0], batch_size)
    return x_train[index, :], y_train[index]


def prune_weights(weight):
    for i in range(weight.shape[-1]):
        tmp = deepcopy(weight[..., i])
        tmp = np.abs(tmp)
        tmp = np.sort(np.array(tmp))
        # compute threshold
        threshold = tmp[int(tmp.shape[0] * COMPRESSION_RATE)]
        weight[..., i][np.abs(weight[..., i]) < threshold] = 0
    sparse_layer = deepcopy(weight)
    sparse_layer[sparse_layer != 0] = 1
    return weight, sparse_layer


Sparse = {}

# Pruning
for layer_id in range(len(model.layers)):
    layer = model.layers[layer_id]
    weight = layer.get_weights()
    # weight:weight[0]
    # bias:weight[1]
    if len(weight) > 0:
        if layer_id != 0:
            w = deepcopy(weight)
            new_weight, sparse_layer = prune_weights(w[0])
            Sparse[layer_id] = sparse_layer
            w[0] = new_weight
            layer.set_weights(w)


score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])

# Retrain
for epoch in range(NUM_EPOCH):
    for j in range(x_train.shape[0] // BATCH_SIZE):
        begin = j*BATCH_SIZE
        if j*BATCH_SIZE + BATCH_SIZE > x_train.shape[0]:
            end = x_train.shape[0]
        else:
            end = j*BATCH_SIZE + BATCH_SIZE
        X, Y = x_train[begin:end], y_train[begin:end]
        # train on each batch
        model.train_on_batch(X, Y)
        # apply Sparse connection
        for layer_id in Sparse:
            w = model.layers[layer_id].get_weights()
            w[0] = w[0] * Sparse[layer_id]
            model.layers[layer_id].set_weights(w)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))


score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])

cluster_index = dict()
cluster_centroids = dict()


# Weight Share and Quantization
for layer_id in Sparse:
    layer = model.layers[layer_id]
    weight = layer.get_weights()
    w = deepcopy(weight)
    shape = w[0].shape
    weight_array = w[0].flatten()
    max_weight = max(weight_array)
    min_weight = min(weight_array)
    space = np.linspace(min_weight, max_weight, num=2 ** BITS)
    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
                    algorithm="full")
    kmeans.fit(weight_array.reshape(-1, 1))
    # cluster index of each weight
    layer_cluster_index = kmeans.labels_
    # value of the centroids
    layer_centroids = kmeans.cluster_centers_.flatten()
    # Add to dict
    cluster_index[layer_id] = layer_cluster_index
    cluster_centroids[layer_id] = layer_centroids
    # set new weight
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(shape)
    w[0] = new_weight
    layer.set_weights(w)


# calculate gradient and get the fine-tuned centroids
for epoch in range(NUM_EPOCH):
    for j in range(x_train.shape[0] // BATCH_SIZE):
        begin = j * BATCH_SIZE
        if j * BATCH_SIZE + BATCH_SIZE > x_train.shape[0]:
            end = x_train.shape[0]
        else:
            end = j * BATCH_SIZE + BATCH_SIZE
        X, Y = x_train[begin:end], y_train[begin:end]
        with tf.GradientTape() as tape:
            y_predict = model(X)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=y_predict)
        grads = tape.gradient(loss, model.variables)
        for layer_id in Sparse:
            gradient = grads[layer_id].numpy().flatten()
            gradient_index = np.zeros(2 ** BITS)
            # Calculate the sum of gradient of the same cluster
            for i in range(len(gradient)):
                gradient_index[cluster_index[layer_id][i]] += gradient[i]
            # Update centroid
            fine_tuned_centroids = cluster_centroids[layer_id]-LEARNING_RATE*gradient_index
            cluster_centroids[layer_id] = fine_tuned_centroids

            w = model.layers[layer_id].get_weights()
            shape = w[0].shape
            new_weight = fine_tuned_centroids[cluster_index[layer_id]].reshape(shape)
            w[0] = new_weight
            model.layers[layer_id].set_weights(w)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))


print('-------------------')
score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])
