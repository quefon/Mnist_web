import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cv2
import os
import os.path
import xlrd
from sklearn import svm
import numpy as np

import csv

csvfile = open('lables.csv')
reader = csv.reader(csvfile)

lables = []
for line in reader:
    tmp = [line[0],line[1]]
    #print tmp
    lables.append(tmp)

csvfile.close()

X = []
y = []
picnum = len(lables)
print "picnum : ",picnum
for i in range(0,picnum):
    img = cv2.imread("image/" + lables[i][0] + '.png', cv2.IMREAD_GRAYSCALE)
    X.append(img)
    y.append(lables[i][1])


print len(X),X[0].shape
print len(y),len(y[0])
# cv2.imshow("Image", X[9990])
# cv2.waitKey (0)
# cv2.destroyAllWindows()


# [A-Z] -> [0-25] -> onehot 104 dim's 01 vector(4*26)

labeldict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
num_classes = 10

X = np.array(X)
for i in xrange(len(y)):
    c=[]

    c0 = keras.utils.to_categorical(labeldict[y[i][0]], num_classes)
    c1 = keras.utils.to_categorical(labeldict[y[i][1]], num_classes)
    c2 = keras.utils.to_categorical(labeldict[y[i][2]], num_classes)
    c3 = keras.utils.to_categorical(labeldict[y[i][3]], num_classes)
    c = np.concatenate((c0,c1,c2,c3),axis=0)
    y[i] = c

y = np.array(y)
#y = y[:,0,:]
print "x_shape = ",X.shape
print "y_shape = ",y.shape
print y[:2]

batch_size = 25
epochs = 60

img_rows, img_cols = 60, 160

x_train = X[:20000]
y_train = y[:20000]
x_test = X[20000:]
y_test = y[20000:]

print K.image_data_format()
print x_train.shape,x_test.shape
print x_train.shape[0]
print img_rows,img_cols

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = 255 - x_train
x_test = 255 - x_test
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print lables[:2]
print y_train[:2]


model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 9),activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 4)))

model.add(Conv2D(16, kernel_size=(5, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 3)))

model.add(Flatten())

model.add(Dense(num_classes*4, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#model save
model.save('Model/my_model3.h5')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


