import tensorflow as tf
import numpy
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape, Lambda, BatchNormalization

model = Sequential()

model.add(Reshape((66, 200, 3), input_shape=(66, 200, 3)))

# 3@66x200 -> 24@31x98
model.add(Conv2D(filters = 24,
                 kernel_size = 5,
                 strides = 2,
                 activation="relu",
                 data_format="channels_last",
                 kernel_initializer='random_uniform',
                 padding="valid",
                 input_shape=(66, 200, 3),
                 ))

model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one'))

# 24@31x98 -> 36@14x47
model.add(Conv2D(filters = 36,
                 kernel_size = 5,
                 strides = 2,
                 activation="relu",
                 data_format="channels_last",
                 kernel_initializer='random_uniform',
                 padding="valid",
                 ))

model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one'))

# 36@14x47 -> 48@5x22
model.add(Conv2D(filters = 48,
                 kernel_size = 5,
                 strides = 2,
                 activation="relu",
                 data_format="channels_last",
                 kernel_initializer='random_uniform',
                 padding="valid"
                 ))

model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one'))

# 48@5x22 -> 64@3x20
model.add(Conv2D(filters = 64,
                 kernel_size = 3,
                 strides = 1,
                 activation="relu",
                 data_format="channels_last",
                 kernel_initializer='random_uniform',
                 padding="valid"
                 ))

model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one'))

# 64@3x20 -> 64@1x18
model.add(Conv2D(filters = 64,
                 kernel_size = 3,
                 strides = 1,
                 activation="relu",
                 data_format="channels_last",
                 kernel_initializer='random_uniform',
                 padding="valid"
                 ))

model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one'))

model.add(Flatten())

# 64*1*18 = 1164 -> 100
model.add(Dense(100,
                activation="relu",
                kernel_initializer='random_uniform',
                input_dim=1164))
# 100 -> 50
model.add(Dense(50,
                activation="relu",
                kernel_initializer='random_uniform'))

# 50 -> 10
model.add(Dense(10,
                activation="relu",
                kernel_initializer='random_uniform'))

model.add(Dense(1))


# original loss:
# loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
# avg(output - ground truth)^2 + sum(nn.l2_loss(v)) * 0.001

model.compile(optimizer='adam',
              loss='msle',
              metrics=['acc'])
#              metrics=['accuracy'])
