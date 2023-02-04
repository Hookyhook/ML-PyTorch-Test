import tensorflow as tf

# Define the input and output data
x = tf.constant([[1.4, 2.0, 3.0]])
y = tf.constant([[1.0, 1.234, 1.0]])

# Define a linear regression model using matrix multiplication
model = tf.linalg.matmul(x, tf.transpose(y))

# Evaluate the model to get the result
with tf.Session() as sess:
    result = sess.run(model)
    print("Following results"+result)
