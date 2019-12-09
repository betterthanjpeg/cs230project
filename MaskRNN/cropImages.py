import os
import sys
import random
import math
import numpy as np
import time
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from os import listdir, walk, rename
from os.path import isfile, join

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

import coco
from samples.coco import coco

from PIL import Image

GPU_COUNT = 1
IMAGES_PER_BATCH = 1

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = GPU_COUNT
    IMAGES_PER_GPU = IMAGES_PER_BATCH

def crop_image(img, mask, box):
    x1,y1,x2,y2 = box
    #for j in range(3): mask[:,:,j] = np.array(mask).astype(float)
    reverse_mask = 255*(1-mask).astype(int)
    cropped_image = np.multiply(img, mask).astype(int) + reverse_mask
    return cropped_image

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = '/mnt/d/projects/cs230/presentation_set/raw/'
COMPLETED_DIR = '/mnt/d/projects/cs230/presentation_set/raw/'
EXPORT_DIR = '/mnt/d/projects/cs230/presentation_set/body/'
DISCARD_DIR = '/mnt/d/projects/cs230/disney_images/discard/'

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
count = len(next(os.walk(EXPORT_DIR))[2])
BATCH_SIZE = IMAGES_PER_BATCH * GPU_COUNT
N_BATCHES = int(np.floor(len(file_names) / BATCH_SIZE))
start_time = time.time()

print(f"Num files: {len(file_names)}")

#fi = np.random.choice(file_names, IMAGES_PER_BATCH, replace = False)
for BATCH_NUM in range(N_BATCHES):
	if BATCH_NUM % 100 == 0:
		print(f"Currently on batch num {BATCH_NUM}/{N_BATCHES}")
	fi = file_names[BATCH_NUM * BATCH_SIZE: (BATCH_NUM+1) * BATCH_SIZE]
	#fi = np.random.choice(file_names, BATCH_SIZE, replace = False)
	images = []
	nextFlag = 0
	for fn in fi:
		try:
			image = skimage.io.imread(os.path.join(IMAGE_DIR, fn))
			#image = skimage.transform.resize(image, (256,256))
		except:
			print(f"{fn} already completed")
			nextFlag = 1
			continue
		images.append(image)
		f_src = join(IMAGE_DIR, fn)
		rename(f_src, COMPLETED_DIR + fn)
	if nextFlag == 1: continue

	# Run detection
	try:
		results = model.detect(images, verbose=1)
	except:
		print(f"Failed at detecting {fn}")
		continue

	for idx, r in enumerate(results):

		#print(r)
		image = images[idx]
		idx_persons = np.where((r['class_ids']) == 1)[0]
		n_persons = len(idx_persons)
		masks = r['masks']


		for n in range(n_persons):
		    i = idx_persons[n]

		    if r['scores'][i] < 0.95: continue
		    y1,x1,y2,x2 = r['rois'][i]
		    mask = np.zeros((y2-y1,x2-x1,3))
		    for j in range(3): mask[:,:,j] = np.array(masks[y1:y2,x1:x2,i]).astype(float)

		    cropped_image = crop_image(image[y1:y2,x1:x2,:], mask, [y1,x1,y2,x2]).astype(np.uint8)
		    im = Image.fromarray(cropped_image)
		    print(type(cropped_image))
		    im.save(EXPORT_DIR + 'real' + str(count) + '.jpg')
		    count += 1


print(f"Elapsed time: {time.time() - start_time} secs")