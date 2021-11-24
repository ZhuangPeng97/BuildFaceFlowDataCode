import os
import numpy as np
import struct
import math
import concurrent.futures
import cv2
import random
import pickle
import glob
import sys
import shutil
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import autoinit
from multiprocessing.dummy import Pool as ThreadPool
from skimage.measure import label

# def fillholes(gray):
# 	des = cv2.bitwise_not(gray)
# 	cv2.imwrite("test1.png", des)
# 	contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# 	print(len(contour))
# 	for cnt in contour:
# 		cv2.drawContours(des,[cnt],0,255,-1)

# 	gray = cv2.bitwise_not(des)
# 	cv2.imwrite("test.png", gray)
# 	exit(0)
# 	return gray

def fillHole(im_in):
	# cv2.imwrite("test1.png", im_in)
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv
	# cv2.imwrite("test.png", im_out)
	# exit(0)
	return im_out

def largestConnectComponent(bw_img, ):
	'''
	compute largest Connect component of a binary image

	Parameters:
	---

	bw_img: ndarray
		binary image

	Returns:
	---

	lcc: ndarray
		largest connect component.

	Example:
	---
		>>> lcc = largestConnectComponent(bw_img)

	'''
	labeled_img, num = label(bw_img, connectivity=2, background=0, return_num=True)    
	# plt.figure(), plt.imshow(labeled_img, 'gray')
	max_label = 0
	max_num = 0
	for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
		if np.sum(labeled_img == i) > max_num:
			max_num = np.sum(labeled_img == i)
			max_label = i
	lcc = (labeled_img == max_label)

	return lcc.astype(np.uint8) * 255

def FilterAndSave(mask, color, new_color_path, new_mask_path):
	# min_depth = depth[(mask!=0)].min()
	# mean_depth = depth[(mask!=0)].mean()
	# thre = min_depth + 2.5 * (mean_depth - min_depth)
	# print("thre: ", thre)
	print(color.shape)
	new_mask = mask.copy()
	new_mask = (new_mask!=0).astype(np.uint8)
	new_mask = largestConnectComponent(new_mask)
	#new_mask = fillHole(new_mask)
	new_color = cv2.bitwise_and(color,color,mask=new_mask)
	cv2.imwrite(new_color_path, new_color)
	cv2.imwrite(new_mask_path, new_mask)
	# cv2.imwrite(new_mask_path[:-4]+"1.png", mask)
	return new_mask, new_color

def select_part(mask):
	mask_need = (mask!=0) & (mask!=18)
	mask[mask_need] = 255
	mask[~mask_need] = 0
	return mask

if __name__ == '__main__':
	H = 640
	W = 480
	save_root = "/disk2/zhuang/UNNRT_fixed_BSW/video2img/images_mask_test/"
	if not os.path.exists(save_root):
		os.makedirs(save_root)
	label_root = "/disk2/zhuang/CelebAMask-HQ/face_parsing/test_results1/"
	img_paths = glob.glob(os.path.join("/disk2/zhuang/UNNRT_fixed_BSW/video2img/images_test/", "*.png"))
	random.shuffle(img_paths)
	for path in img_paths:
		mask_path = label_root + os.path.basename(path)[:-4] + ".png"
		print(mask_path)
		if not os.path.exists(mask_path):
			continue
		mask = cv2.imread(mask_path, -1)
		mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
		mask = select_part(mask)
		new_mask_path = save_root + os.path.basename(path)[:-4] + ".png"
		new_color_path = save_root + os.path.basename(path)
		color = cv2.imread(path)
		new_mask, new_color = FilterAndSave(mask, color, new_color_path, new_mask_path)


