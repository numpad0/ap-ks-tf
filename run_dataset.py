from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc
import numpy
import threading
import model
import cv2

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

xf = []
ys = []


with open("driving_dataset/data.txt") as f:
    for line in f:
        xf.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        # TODO: add oversteer/understeer correction
        ys.append(float(line.split()[1]) * scipy.pi / 180)

model = load_model('ap-model.hdf5')

print(model.summary())

frame = numpy.array([0])
i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread(xf[i], mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model((0, (image[0], image[1], image[2])))
    print(xf[i] + " " + str(degrees))
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
