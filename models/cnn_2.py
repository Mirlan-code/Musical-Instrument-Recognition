import tensorflow as tf
import numpy as np
import os

import itertools 
from models.evaluate_model import evaluate_model
from dataset_preparation.preprocess_data import Dataset
from sklearn.model_selection import train_test_split

LABELS_TO_NUMBERS = {
	"bas" : 0,
    "gac" : 1,
    "key" : 2,
    "org" : 3
}


def yololike_model(features, labels, mode):
	"""YOLO-like CNN architecture.
	Parameters:
		features: the array containing the examples used for training
		labels: the array containing the labels of the examples in one-hot representation
		mode: a tf.estimator.ModeKeys like tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
	"""

	# Input Layer
	input_layer = tf.reshape(features, [-1, features.shape[1], features.shape[2], 1])
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=64,
		kernel_size=[7, 7],
		strides=2,
		padding="same",
		activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=192,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=128,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	conv4 = tf.layers.conv2d(
		inputs=conv3,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	conv5 = tf.layers.conv2d(
		inputs=conv4,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	conv6 = tf.layers.conv2d(
		inputs=conv5,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

	conv7 = tf.layers.conv2d(
		inputs=pool3,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv8 = tf.layers.conv2d(
		inputs=conv7,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv9 = tf.layers.conv2d(
		inputs=conv8,
		filters=1024,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	pool4 = tf.layers.max_pooling2d(inputs=conv9, pool_size=[2, 2], strides=2)


	conv10 = tf.layers.conv2d(
		inputs=pool4,
		filters=1024,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print('conv10', conv10.shape)

	pool4_flat = tf.reshape(conv10, [-1, np.prod(conv10.shape[1:])])
	dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=labels.shape[1])

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=tf.argmax(input=labels, axis=1),
			predictions=predictions["classes"]
		)
	}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def cnn(dataset, features):
    instruments = 4
    X, Y = dataset(features, False)
    print(X.shape)
    print(Y.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2.0/10)
    # Y_train = np.zeros((len(Y_traintmp), instruments), dtype=np.float32)
    # Y_test = np.zeros((len(Y_testtmp), instruments), dtype=np.float32)
    for i in range(len(Y_train)):
	    label = Y_train[i]
	    Y_train[i] = LABELS_TO_NUMBERS[label]
	
    for i in range(len(Y_test)):
	    label = Y_test[i]
	    Y_test[i] = LABELS_TO_NUMBERS[label]
		
    # for i in range(len(Y_traintmp)):
    #     label = Y_traintmp[i]
    #     Y_train[i][LABELS_TO_NUMBERS[label]] = 1

    # for i in range(len(Y_testtmp)):
    #     label = Y_traintmp[i]
    #     Y_test[i][LABELS_TO_NUMBERS[label]] = 1

    saved_model_path = os.getcwd() + '/cnn_models/yolo-mel'
    feature_size = X_train.shape[1]
    print(saved_model_path)
    print(X_train.shape)
    print(Y_train.shape)

    X_train = X_train.reshape(1900, feature_size, 1)
    X_test = X_test.reshape(len(X_test), feature_size, 1)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, input_shape = (feature_size, 1), activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=200, activation='relu'),

        tf.keras.layers.Dense(units=4, activation='softmax')
    ])
    model.compile('adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    print(model.summary())
    model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
    
    

def findsubsets(s, n): 
    return [set(i) for i in itertools.combinations(s, n)]


if __name__ == "__main__":
    dataset = Dataset(path="dataset", reinitialize=False)
    s = [
        # "chroma_stft",
        # "chroma_cqt",
        # "chroma_cens",
        # "melspectogram",
        "rms",
        # "spectral_centroid",
        # "spectral_bandwidth",
        # "spectral_contrast",
        # "spectral_flatness",
        # "spectral_rolloff",
        # "poly_features",
        # "tonnetz",
        # "zero_crossing_rate",

        "tempogram",
        # "fourier_tempogram",
        # "track"
    ]
    answer = []
    for i in range(len(s), len(s) + 1):
        lists = findsubsets(s, i)
        for l in lists:
            print(l)
            res = cnn(dataset, l)
            answer.append([l, res])
    answer = sorted(answer, key=lambda res: res[1], reverse=True)
    for i in answer:
        print(i[0])
        print(str(i[1]))