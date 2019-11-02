from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image, ImageFilter
import sys
import os
import csv

MAX = 30000
pix_l =[]
width=120
height=36
train_dir="image"

def gen_image(name):
	img = ImageCaptcha()
    image = img.generate_image(name)
	#image = img.create_captcha_image(name, (255,0,0), (255,255,255))
	#image = depoint(image)
	#image = binaring(image)
	image.save(train_dir + '/' + name + '.png')
        with open('lables.csv', 'a') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow([name, name])

def text():
	for i in range (MAX+1):
		name = i
		if(name<=MAX):
			name = str(name)
			name = name.zfill(4)
		else:
			name = str(name)
		gen_image(name)

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

text()

