from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import scipy.misc
import numpy
import threading
import cv2
import time

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

xf = []
xi = []
ys = []
frame = []

idg = ImageDataGenerator(width_shift_range = 0.05, height_shift_range = 0.05)

batch_size = 100
epochs = 30

with open("driving_dataset/data.txt") as f:
    for line in f:
        xf.append("driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * scipy.pi / 180)

model = load_model('ap-model.hdf5')

print(model.summary())

get_last_output = K.function([model.layers[0].input, K.learning_phase()],
                        [model.layers[13].output])

get_1st_output = K.function([model.layers[0].input, K.learning_phase()],
                        [model.layers[1].output])
i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread(xf[i], mode="RGB")
    image = numpy.expand_dims((scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0), axis=0)
    print(image.shape)
    degrees = get_last_output([image, 0])
    output_1st = numpy.empty((31, 392), dtype=float)
    layers = get_1st_output([image, 0])[0]
    layers = numpy.rollaxis(layers, 3, 1)
    for p in range(1, 6):
        output_1st_row = numpy.empty((31, 98), dtype=float)
        for q in range (0, 3):
            output_1st_row = numpy.hstack((output_1st_row, layers[0][p*q]))
        output_1st = numpy.vstack((output_1st, output_1st_row))
    print("output: ", degrees)
    print("truth : ", ys[i])
    smoothed_angle = (float(degrees[0]))
    M = cv2.getRotationMatrix2D((cols/2,rows/2),smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("1st layer output", output_1st)
    cv2.imshow("steering wheel", dst)
    time.sleep(0.01)
    i += 1
cv2.destroyAllWindows()
