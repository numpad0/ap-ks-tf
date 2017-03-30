from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc
import numpy
import threading
import model
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

def generate_arrays_from_file(arg):
    global frame
    counter = 0
    ximg = []
    yval = []
    while 1:
        with open("driving_dataset/data.txt") as f:
            for line in f:
                path = ("driving_dataset/" + line.split()[0])
                dimg = scipy.misc.imread(path)
                frame = dimg
                ximg.append(scipy.misc.imresize(dimg, [66, 200])/ 255.0)
                yval.append(float(line.split()[1]) * scipy.pi / 180)
                counter += 1
                if(counter >= arg):
                    yield (numpy.asarray(ximg), yval)
                    counter = 0
                    ximg = []
                    yval = []

model = load_model('ap-model.hdf5')

print(model.summary())

gen = generate_arrays_from_file(1)

while(cv2.waitKey(10) != ord('q')):
    degrees = model.predict_generator(gen, steps=1, verbose=1)
    print(degrees[0][0])
    smoothed_angle = degrees[0][0]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imshow("steering wheel", dst)
    time.sleep(0.05)
cv2.destroyAllWindows()
