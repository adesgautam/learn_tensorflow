
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# import utils

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Data Placeholders
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Variables
W = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Model
Y_predicted = tf.add(tf.multiply(W,X), b)

# Squared Loss
# loss = tf.square(Y-Y_predicted, name='loss')

# Huber Loss
def huber_loss(labels, predictions, delta=1.0):
	residual = tf.abs(predictions-labels)
	condition = tf.less(residual, delta)
	small_res = 0.5*tf.square(residual)
	large_res = delta*residual - 0.5*tf.square(delta)
	return tf.where(condition, small_res, large_res)

loss = huber_loss(Y, Y_predicted)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs/linear_regression', sess.graph)

	for i in range(100):
		total_loss = 0
		for x, y in data:
			_, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
			total_loss += l

		print('Epoch: {0} Loss: {1}'.format(i, total_loss/n_samples))

	writer.close()

	w, b = sess.run([W, b])

X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label="Real Data")
plt.plot(X, w*X+b, 'r', label="Predicted Data")
plt.legend()
plt.show()

