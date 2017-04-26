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
        ys.append(float(line.split()[1]) / 576)
        # assuming 1152 degrees lock-to-lock to map input to -1 to 1

model = load_model('ap-model.hdf5')

print(model.summary())
model_layer_count = len(model.layers)
print("layer count: ", model_layer_count)
get_last_output = K.function([model.layers[0].input, K.learning_phase()],
                        [model.layers[model_layer_count-1].output])

get_1st_output = K.function([model.layers[0].input, K.learning_phase()],
                        [model.layers[1].output])
get_2nd_output = K.function([model.layers[0].input, K.learning_phase()],
                        [model.layers[2].output])
i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread(xf[i], mode="RGB")
    image = numpy.expand_dims((scipy.misc.imresize(full_image, [66, 200]) / 255.0), axis=0)
    degrees = get_last_output([image, 0])
    output_1st = numpy.empty((31, 392), dtype=float) #24@31x98
    output_2nd = numpy.empty((14, 282), dtype=float) #36@14x47
    layers_1st = get_1st_output([image, 0])[0]
    layers_1st = numpy.rollaxis(layers_1st, 3, 1)
    layers_2nd = get_2nd_output([image, 0])[0]
    layers_2nd = numpy.rollaxis(layers_2nd, 3, 1)
    for p in range(1, 6):
        output_1st_row = numpy.empty((31, 98), dtype=float)
        for q in range (0, 3):
            output_1st_row = numpy.hstack((output_1st_row, layers_1st[0][p*q]))
        output_1st = numpy.vstack((output_1st, output_1st_row))

    for r in range(1, 6):
        output_2nd_row = numpy.empty((14, 47), dtype=float)
        for s in range (0, 5):
            output_2nd_row = numpy.hstack((output_2nd_row, layers_2nd[0][r*s]))
        output_2nd = numpy.vstack((output_2nd, output_2nd_row))
    out = "{:.7f}".format(degrees[0][0][0])
    truth = "{:.7f}".format(ys[i])
    perc = (((degrees[0][0][0] - (ys[i]))/(ys[i]))*100)
    print("output: ", out, "truth: ", truth, "precision: ", perc)
    smoothed_angle = (float(degrees[0][0][0]) * 1152)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("1st layer output", output_1st)
    cv2.imshow("2nd layer output", output_2nd)
    cv2.imshow("steering wheel", dst)
    time.sleep(0.01)
    i += 1
cv2.destroyAllWindows()
