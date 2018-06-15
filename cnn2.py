import tensorflow as tf


a = tf.truncated_normal([16, 128, 128, 3])
session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(tf.shape(a))

classes = ['human_face', 'undef']
num_classes = len(classes)

train_path = 'abc'

validation_size = 0.2
batch_size = 16

data =