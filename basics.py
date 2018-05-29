import tensorflow as tf

x = 5
y = 8
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op1, op2)

# Constants
a = tf.constant([2,2], name='a')
b = tf.constant([[0,1], [2,3]], name='b')
op4 = tf.add(a, b, name="add")
op5 = tf.multiply(a, b, name="mul")

# more functions
z = tf.zeros([2,3], tf.int32)
z_like = tf.zeros_like(z)
o = tf.ones([2,4], tf.float32)
o_like = tf.ones_like(o)
fill = tf.fill([3,3], 3)

# Constants as sequences
l = tf.linspace(10.0, 20.0, 5) # 20.0 will be included
r = tf.range(3.0, 18.0, 3) # 18.0 will not be included

ops = [op1, op2, op3, op4, op5]

with tf.Session() as sess:
    # start tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    all_ops = sess.run(ops)
    print(all_ops)

writer.close()

# # Explicitly define graph (User defined graphs)
# g = tf.Graph()
# with g.as_default():
#     a = tf.add(3, 5)

# sess = tf.Session(graph=g)
# print(sess.run(a))
# sess.close()

# # Handle default graph
# g = tf.get_default_graph()




