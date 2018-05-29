import tensorflow as tf

tf.set_random_seed(7)

# Normal Distribution
r_n = tf.random_normal((2,2), mean=0.0, stddev=1.0, dtype=tf.float32)

t_n = tf.truncated_normal((2,3), mean=0.0, stddev=1.0, dtype=tf.float32)

# Uniform Distribution
r_u = tf.random_uniform((2,3), minval=0, maxval=20, dtype=tf.float32)

# Randomly shuffle
a = tf.constant([[1,2], [3,4], [4,5]])
r_s = tf.random_shuffle(a)

# Random Crop
r_c = tf.random_crop(a, [2,2])

ops = [r_n, t_n, r_u, r_s, r_c]

with tf.Session() as sess:
    # start tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    all_ops = sess.run(ops)
    print("Random Normal: ", all_ops[0])
    print("Truncated Normal: ", all_ops[1])
    print("Random Uniform: ", all_ops[2])
    print("Random Shuffle: ", all_ops[3])
    print("Random Crop: ", all_ops[4])

writer.close()
