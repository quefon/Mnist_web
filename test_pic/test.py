import os
import random
from captcha.image import ImageCaptcha

for i in range (10):
	name = random.randint(0,1000)

	if(name<1000):
		name = str(name)
		name = name.zfill(4)
	else:
		name = str(name)

	img = ImageCaptcha()
	image = img.generate_image(name)
	image.save(name + ".png")
