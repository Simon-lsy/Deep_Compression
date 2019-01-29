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
BITS = 3


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

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=50)
score = model.evaluate(x_test, y_test)
print(score[1])

model.save_weights('./result/my_model.h5', save_format='h5')

# model.load_weights('./result/my_model.h5')
# score = model.evaluate(x_test, y_test)
# print(score[1])
#
#
#
#
# def prune_weights(weight):
#     for i in range(weight.shape[-1]):
#         tmp = deepcopy(weight[..., i])
#         tmp = np.abs(tmp)
#         # tmp = tmp[tmp >= 0]
#         tmp = np.sort(np.array(tmp))
#         # compute threshold
#         threshold = tmp[int(tmp.shape[0] * COMPRESSION_RATE)]
#         weight[..., i][np.abs(weight[..., i]) < threshold] = 0
#     sparse_layer = deepcopy(weight)
#     sparse_layer[sparse_layer != 0] = 1
#     return weight, sparse_layer
#
#
# Sparse = {}
#
# # Pruning
# for layer_id in range(len(model.layers)):
#     layer = model.layers[layer_id]
#     weight = layer.get_weights()
#     # weight:weight[0]
#     # bias:weight[1]
#     if len(weight) > 0:
#         if layer_id != 0:
#             w = deepcopy(weight)
#             new_weight, sparse_layer = prune_weights(w[0])
#             Sparse[layer_id] = sparse_layer
#             w[0] = new_weight
#             layer.set_weights(w)
#
# # score = model.evaluate(x_test, y_test, verbose=0)
# # print(score[1])
#
# # Retrain
# # for i in range(3):
# #     for j in range(x_train.shape[0] // BATCH_SIZE):
# #         begin = j*BATCH_SIZE
# #         if j*BATCH_SIZE + BATCH_SIZE > x_train.shape[0]:
# #             end = x_train.shape[0]
# #         else:
# #             end = j*BATCH_SIZE + BATCH_SIZE
# #         X, Y = x_train[begin:end], y_train[begin:end]
# #         # train on each batch
# #         model.train_on_batch(X, Y)
# #         # apply Sparse connection
# #         for layer_id in Sparse:
# #             w = model.layers[layer_id].get_weights()
# #             w[0] = w[0] * Sparse[layer_id]
# #             model.layers[layer_id].set_weights(w)
# #     score = model.evaluate(x_test, y_test, verbose=0)
# #     print('val loss: {}'.format(score[0]))
# #     print('val acc: {}'.format(score[1]))
#
#
# # print(layer_count)
# score = model.evaluate(x_test, y_test, verbose=0)
# print(score[1])

# def loss(model,x,y):
#     y_=model(x.astype(np.float32))
#     return tf.losses.softmax_cross_entropy(onehot_labels=y,logits=y_)
#
# #返回一个梯度对象
# def grad(model,inputs,targets):
#     with tf.GradientTape() as tape:
#         loss_value=loss(model,inputs,targets)
#     return tape.gradient(loss_value,model.variables)    #返回梯度对象，传入损失函数和优化对象作为构造函数的参数
#
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#
#
# for j in range(x_train.shape[0] // BATCH_SIZE):
#     begin = j * BATCH_SIZE
#     if j * BATCH_SIZE + BATCH_SIZE > x_train.shape[0]:
#         end = x_train.shape[0]
#     else:
#         end = j * BATCH_SIZE + BATCH_SIZE
#     X, Y = x_train[begin:end], y_train[begin:end]
#     grads = grad(model, X, Y)
#     # print(grads)


#
# # Weight Share and Quantization
# for layer in model.layers:
#     weight = layer.get_weights()
#     if len(weight) > 0:
#         w = deepcopy(weight)
#         shape = w[0].shape
#         weight_array = w[0].flatten()
#         max_weight = max(weight_array)
#         min_weight = min(weight_array)
#         # print(len(weight_array))
#         space = np.linspace(min_weight, max_weight, num=2 ** 5)
#         # print(space)
#         if len(weight_array) < 1500:
#             kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
#                             algorithm="full")
#             kmeans.fit(weight_array.reshape(-1, 1))
#             cluster_index = kmeans.labels_.reshape(shape)
#             # print(cluster_index)
#             # print(kmeans.cluster_centers_)
#             new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(shape)
#             w[0] = new_weight
#             layer.set_weights(w)
#             # get gradient
#         for i in range(3):
#             for j in range(x_train.shape[0] // BATCH_SIZE):
#                 begin = j * BATCH_SIZE
#                 if j * BATCH_SIZE + BATCH_SIZE > x_train.shape[0]:
#                     end = x_train.shape[0]
#                 else:
#                     end = j * BATCH_SIZE + BATCH_SIZE
#                 X, Y = x_train[begin:end], y_train[begin:end]
#                 # print(X[0])
#                 with tf.GradientTape() as tape:
#                     y_predict = model.predict(X)
#                     y_logit_pred = model(X.astype(np.float32))
#                     print(y_predict[0])
#                     print(y_logit_pred[0])
#                     # y_predict = tf.convert_to_tensor(y_predict)
#                     # y_predict = model.predict_classes(X)
#                     # print(y_predict)
#                     # y_predict = keras.utils.to_categorical(y_predict, 10)
#                     # loss = tf.reduce_mean(tf.keras.metrics.categorical_crossentropy(Y, y_predict))
#                     loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=y_predict)
#                     # loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_predict)
#                     print(loss)
#                     # print(loss.numpy())
#                 # print(model.variables)
#                 grads = tape.gradient(loss, model.variables)
#                 print(grads)
#
#
# print('-------------------')
# score = model.evaluate(x_test, y_test, verbose=0)
# print(score[1])
