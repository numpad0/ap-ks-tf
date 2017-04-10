from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc
import numpy
import random
import threading
import model

xf = []
xi = []
ys = []
fl = []

with open("driving_dataset/data.txt") as f:
    for line in f:
        fl.append(line)

random.shuffle(fl)

batch_size = 100
epochs = 10

def generate_arrays_from_file(arg):
    counter = 0
    ximg = []
    yval = []
    while 1:
        for line in fl:
            path = ("driving_dataset/" + line.split()[0])
            ximg.append(scipy.misc.imresize(scipy.misc.imread(path), [66, 200])/ 255.0)
            yval.append(float(line.split()[1]) / 576)
            counter += 1
            if(counter >= arg):
                yield (numpy.asarray(ximg), yval)
                counter = 0
                ximg = []
                yval = []

model = model.model

print(model.summary())

gen = generate_arrays_from_file(batch_size)
model.fit_generator(gen,
                    steps_per_epoch = 100,
                    epochs = epochs,
                    max_q_size = 1000,
                    verbose=1,
                    validation_data = generate_arrays_from_file(20),
                    validation_steps = 20)

# model.load_weights('ap-params.hdf5')
model.save('ap-model.hdf5')
