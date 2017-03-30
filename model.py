from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape

model = Sequential()

model.add(Reshape((66, 200, 3), input_shape=(66, 200, 3)))

# 3@66x200 -> 24@31x98
model.add(Conv2D(filters = 24,
                 kernel_size = 5,
                 strides = 2,
                 activation="relu",
                 data_format="channels_last",
                 padding="valid",
                 input_shape=(66, 200, 3),
                 ))

# 24@31x98 -> 36@14x47
model.add(Conv2D(filters = 36,
                 kernel_size = 5,
                 strides = 2,
                 activation="relu",
                 data_format="channels_last",
                 padding="valid"
                 ))

# 36@14x47 -> 48@5x22
model.add(Conv2D(filters = 48,
                 kernel_size = 5,
                 strides = 2,
                 activation="relu",
                 data_format="channels_last",
                 padding="valid"
                 ))

# 48@5x22 -> 64@3x20
model.add(Conv2D(filters = 64,
                 kernel_size = 3,
                 strides = 1,
                 activation="relu",
                 data_format="channels_last",
                 padding="valid"
                 ))

# 64@3x20 -> 64@1x18
model.add(Conv2D(filters = 64,
                 kernel_size = 3,
                 strides = 1,
                 activation="relu",
                 data_format="channels_last",
                 padding="valid"
                 ))

model.add(Flatten())

# 64*1*18 = 1164 -> 100
model.add(Dense(100, input_dim=1164))

# 100 -> 50
model.add(Dense(50))

# 50 -> 10
model.add(Dense(10))

model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])