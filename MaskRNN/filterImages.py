import os
import sys
import random
import math
import numpy as np
import time
import skimage
import skimage.io
import skimage.filters
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
from os import listdir, walk, rename
from os.path import isfile, join
from PIL import Image


IMAGE_DIR = '/mnt/d/projects/cs230/disney_images_cropped/'
COMPLETED_DIR = '/mnt/d/projects/cs230/disney_images_cropped/'
EXPORT_DIR = '/mnt/d/projects/cs230/disney_images_cropped_filtered/'

file_names = next(os.walk(IMAGE_DIR))[2]
count = len(next(os.walk(EXPORT_DIR))[2])
N_BATCHES = int(np.floor(len(file_names)))

start_time = time.time()
fi = np.random.choice(file_names, N_BATCHES, replace = False)

for BATCH_NUM in range(N_BATCHES):
	if BATCH_NUM % 100 == 0:
		print(f"Currently on batch num {BATCH_NUM}/{N_BATCHES}")
	#fi = file_names[BATCH_NUM * BATCH_SIZE: (BATCH_NUM+1) * BATCH_SIZE]
	fn = fi[BATCH_NUM]
	nextFlag = 0

	try:
		image = skimage.color.rgb2gray(skimage.io.imread(os.path.join(IMAGE_DIR, fn)))
	except:
		print(f"{fn} already completed")
		nextFlag = 1
		continue

	filt_real, filt_image = skimage.filters.gabor(image, frequency = 0.8)
	filt_image = skimage.filters.sobel(image)
	filt_image = np.absolute(filt_image)
	max_pix = np.max(filt_image)
	filt_image /= max_pix
	filt_image[filt_image>1] = 1
	filt_image[filt_image<0.05] = 0

	p2, p98 = np.percentile(filt_image[filt_image < 0.999], (2, 98))
	filt_image = skimage.exposure.rescale_intensity(filt_image, in_range=(p2, p98))

	filt_image_rgb = np.round(skimage.color.gray2rgb(filt_image) * 255).astype(np.uint8)

	f_src = join(IMAGE_DIR, fn)
	rename(f_src, COMPLETED_DIR + fn)
	if nextFlag == 1: continue

	im = Image.fromarray(filt_image_rgb)
	im.save(EXPORT_DIR + fn)
	count += 1