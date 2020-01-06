from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

try:
  tf.enable_eager_execution()
  print('Eager execution enabled')
except ValueError:
  print('Already running in Eager mode')

tfe = tf.contrib.eager

# Define matrix A
A = np.array(
  [[1.0, 3.0],
   [2.0, 1.0],
   [4.0, 2.0]]
)

# Define matrix B
B = np.array(
  [[6.0, 2.0, 1.0],
   [3.0, 4.0, 5.0]]
)

# Define vector x
x = np.array([3.0, 2.0])

print('A.shape is:', A.shape, 'B.shape is:', B.shape, 'x.shape is:', x.shape)

# Using numpy dot
y = A.dot(x)

print('Using dot:\t y =', y, '\t y.shape =', y.shape)

# Using einsum
y = np.einsum('ij, j', A, x)

print('Using einsum:\t y =', y, '\t y.shape =', y.shape)

# Manual version 1
y = np.array([
    A[0,0] * x[0] + A[0,1] * x[1],
    A[1,0] * x[0] + A[1,1] * x[1],
    A[2,0] * x[0] + A[2,1] * x[1],
    ])
print('Manual 1:\t y =', y, '\t y.shape =', y.shape)

# Manual version 2:
# Matrix-vector multiplication can be thought of as a linear combination of the columns of of the matrix
y = x[0] * A[:,0]  +  x[1] * A[:, 1]

print('Manual 2:\t y =', y, '\t y.shape =', y.shape)

# Using numpy dot
C = A.dot(B)

print('Using DOT: C= \n\n', C, '\n\nC.shape =', C.shape)

# Using einsum
C = np.einsum('ik, kj', A, B)
print('\n\nUsing einsum: C= \n\n', C, '\n\nC.shape =', C.shape)

# Note, the above einsum notation is equivalent to the following
C = np.einsum('ik, kj -> ij', A, B)

# And in Tensorflow
C = tf.matmul(A, B)
print('\n\nUsing Tensorflow: C= \n\n', C, '\n\nC.shape =', C.shape)

# Matrix multiplication is not commutative:
C = B.dot(A)
print('C: \n', C)
print()
print('C.shape:', C.shape)
