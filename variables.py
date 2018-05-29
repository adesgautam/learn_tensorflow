import tensorflow as tf

a = tf.Variable(2, name='scalar')

b = tf.Variable([2,3], name='vector')

c = tf.Variable([[0,1], [2,3]], name='matrix')

W = tf.Variable(tf.zeros([784,10]))

# Initialize all variables
init = tf.global_variables_initializer()

# Initialize only subset of variales
# init_ab = tf.variables_nitializer([a,b], name='init_ab')

with tf.Session() as sess:
    # start tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(init)

    # Initialize subset of variables
    # sess.run(init_ab)

    # Initialize a single variables
    # sess.run(W.initializer)
    # print(W.eval()) # for printing values of W
    

writer.close()
