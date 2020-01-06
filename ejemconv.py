import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print("TensorFlow executing eagerly: {}".format(tf.executing_eagerly()))
# Create a random colour "image" of shape 10x10 with a depth of 3 (for red, green and blue)
dummy_input = np.random.uniform(size=[10, 10, 3])
fig, ax = plt.subplots(1, 1)
plt.imshow(dummy_input)
ax.grid(False)
print('Input shape: {}'.format(dummy_input.shape))

filters = 1  #@param { type: "slider", min:0, max: 10, step: 1 }
kernel_size = 2 #@param { type: "slider", min:1, max: 10, step: 1 }
stride = 1 #@param { type: "slider", min:1, max: 3, step: 1 }

conv_layer = tf.keras.layers.Conv2D(
    filters=filters,
    kernel_size=kernel_size,
    strides=stride,
    padding="valid",
    input_shape=[10, 10, 3])

# Convert the image to a tensor and add an extra batch dimension which
# the convolutional layer expects.
input_tensor = tf.convert_to_tensor(dummy_input[None, :, :, :])
convoluted = conv_layer(input_tensor)

print('The output dimension is: {}'.format(list(convoluted.shape)[1:]))
print('The number of parameters is: {}'.format(conv_layer.count_params()))

print ('the number of hyperparameter is : {}'.format(list(convoluted.shape)[1:]))
#optional code
X = np.array([[9, 5, 4, 5, 6, 4],
              [6, 6, 3, 5, 8, 2],
              [4, 6, 9, 1, 3, 6],
              [9, 7, 1, 5, 8, 1],
              [4, 9, 9, 5, 7, 3],
              [7, 3, 6, 4, 9, 1]])

max_pool_layer = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
max_pool_layer(tf.convert_to_tensor(X[None, :, :, None])).numpy().squeeze()
