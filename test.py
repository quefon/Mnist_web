from keras.models import load_model
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image, ImageFilter
import os
import random
import cv2

from keras import backend as K

pix_l =[]
img_rows, img_cols = 60, 160

def binaring(image,threshold = 200):
	pixdata = image.load()
	#a = np.array(pixdata)

	#print(pixdata)
	w, h = image.size #default (160,60)
	#print(image.size)
	for y in range(h):
		for x in range(w):
			#sys.stdout.write(str(pixdata[x,y]) + ' ')
			pix_l.append(pixdata[x,y])
			if pixdata[x, y] < threshold:
				pixdata[x, y] = 0
			else:
				pixdata[x, y] = 255

	return image

def depoint(image):
    image = image.convert('L')
    im_f = image.filter(ImageFilter.MedianFilter(size=5))
    im_f = im_f.filter(ImageFilter.MedianFilter(size=5))
    im_f = im_f.filter(ImageFilter.MedianFilter(size=5))
    im_f = im_f.filter(ImageFilter.MedianFilter(size=5))
    return im_f

#generate Four digits random number
name = random.randint(0,1000)

if(name<1000):
	name = str(name)
	name = name.zfill(4)
else:
	name = str(name)

#generate captcha
img = ImageCaptcha()
image = img.generate_image(name)
image.show()
#image = img.create_captcha_image(name, (255,0,0), (255,255,255))
#image = depoint(image)
#image = binaring(image)
#image.show()
image.save(name + ".png")
# x_test
X = []
img = cv2.imread(name + '.png', cv2.IMREAD_GRAYSCALE)
X.append(img)
print len(X),X[0].shape
X = np.array(X)
print "x_shape = ",X.shape
X = 255 - X
X = X.astype('float32')
X /= 255
print(X.shape[0], 'test samples')

# format to dimension four
print "k.image_data_format() = ", K.image_data_format()
if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print "x_shape = ",X.shape



#load model
model = load_model('Model/my_model3.h5')

pred = model.predict(X)
outdict = ['0','1','2','3','4','5','6','7','8','9']

correct_num = 0




for i in range(pred.shape[0]):
	c0 = outdict[np.argmax(pred[i][:10])]
	c1 = outdict[np.argmax(pred[i][10:10*2])]
	c2 = outdict[np.argmax(pred[i][10*2:10*3])]
	c3 = outdict[np.argmax(pred[i][10*3:])]
	c = c0+c1+c2+c3
	#print c,lables[4000+i][1]
	print "captcha number = " +  str(c)
	#print "name = " + str(name)
	if c == name:
		correct_num = 1

if(correct_num):
	print("correct")
else:
	print("Wrong Answer")
#print "Test Whole Accurate : ", float(correct_num)/len(pred)

os.remove(name + '.png')
"""
print('processing...')
prediction = model.predict_classes(letter)
print(prediction)
"""
