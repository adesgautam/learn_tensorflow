import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5,5,5], tf.float32)

c = tf.add(a, b)

with tf.Session() as sess:
    # start tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    
    # initialize a and then c will be evaluated
    res = sess.run(c, {a: [1,2,3]} )
    print(res)

writer.close()
