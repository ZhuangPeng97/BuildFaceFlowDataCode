import os
import numpy as np
import struct
import math
import csv

def ComputeCDDistance(warp_obj_path, tar_obj_path, csv_path):
	try:
		os.system('./build/CLOEST '+warp_obj_path+' '+tar_obj_path+' '+csv_path)
	except Exception as e:
		print("==========Exception:", e)
		print(scr + " and " + tar + " skipped!!!")
	
if __name__ == '__main__':
	data_path = "/disk2/zhuang/reg_data/save_data/results1017/results/"
	data_root = "/disk2/zhuang/reg_data/data/"
	csv_path = "/disk2/zhuang/reg_data/save_data/results1017/chamfer1017.csv"
	with open(csv_path, 'a', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['name', 'src2tar', 'tar2src', 'src2tar_average', 'tar2src_average', 'CD'])
	
	all_file = os.listdir(data_path)
	for f in all_file:
		if f.split('.')[-1] == 'obj':
			warp_obj_path = data_path+f
			f_split = f.split("_")
			tar_obj_path = data_root+f_split[4]+"/"+f_split[5]+"/"+f_split[6]+"/"+f_split[7]
			ComputeCDDistance(warp_obj_path, tar_obj_path, csv_path)


	
