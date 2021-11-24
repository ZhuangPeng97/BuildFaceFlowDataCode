import os
import sys
import numpy as np
import struct
import math
import cv2
import pickle
import logging
from multiprocessing.dummy import Pool as ThreadPool
from skimage.measure import label
import openmesh as om
import shutil

def save_graph(nodes_index, edges_array_new, save_graph_path):
	with open(save_graph_path, 'wb') as fout:
		edge_total_size = 4 * 2 * edges_array_new.shape[0]
		fout.write(struct.pack('I', edge_total_size))
		fout.write(
			struct.pack('={}I'.format(edges_array_new.size),
						*edges_array_new.flatten("C")))
		nodes_size = 4 * nodes_index.shape[0]
		fout.write(struct.pack('I', nodes_size))
		fout.write(
			struct.pack('={}I'.format(nodes_index.size),
						*nodes_index.flatten("C")))


def load_graph(save_graph_path):
	with open(save_graph_path, 'rb') as fin:
		edge_total_size = struct.unpack('I', fin.read(4))[0]
		edges = struct.unpack('I' * (int(edge_total_size / 4)),
							  fin.read(edge_total_size))
		edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2).transpose()
		nodes_total_size = struct.unpack('I', fin.read(4))[0]
		nodes = struct.unpack('I' * (int(nodes_total_size / 4)),
							  fin.read(nodes_total_size))
		nodes = np.asarray(nodes, dtype=np.uint32).reshape(-1)
	return nodes, edges


def show_graph(save_graph_path, save_neighb_path, color, mask):
	nodes, edges = load_graph(save_graph_path)
	with open(save_neighb_path, 'rb') as fin:
		nodes_ids = pickle.load(fin)

	test_img = color.copy()
	for i in range(nodes.shape[0]):
		cv2.circle(test_img, (int(nodes[i] % W), int(nodes[i] / W)), 2,
				   (0, 0, 255))

	for i in range(edges.shape[1]):
		start = (int(nodes[edges[0, i]] % W), int(nodes[edges[0, i]] / W))
		end = (int(nodes[edges[1, i]] % W), int(nodes[edges[1, i]] / W))
		r = np.random.randint(256)
		g = np.random.randint(256)
		b = np.random.randint(256)
		cv2.line(test_img, start, end, (r, g, b), 1)
	if not os.path.exists("debug"):
		os.makedirs("debug")
	cv2.imwrite("./debug/graph.png", test_img)

	test_img = color.copy()
	for i in range(nodes.shape[0]):
		cv2.circle(test_img, (int(nodes[i] % W), int(nodes[i] / W)), 2,
				   (0, 0, 255))
	ex = np.random.randint(0, (mask != 0).sum(), (10, 1))
	non_zero = mask.nonzero()
	for i in range(10):
		start = (non_zero[1][ex[i, 0]], non_zero[0][ex[i, 0]])
		for j in range(num_adja):
			r = np.random.randint(256)
			g = np.random.randint(256)
			b = np.random.randint(256)
			end = (int(nodes[nodes_ids[ex[i, 0], j]] % W),
				   int(nodes[nodes_ids[ex[i, 0], j]] / W))
			cv2.line(test_img, start, end, (r, g, b), 2)
	cv2.imwrite("./debug/neigh_ids.png", test_img)


def create_obj(save_obj_path, depth, mask):
	mesh = om.TriMesh()
	vh_list = []
	for i in range(H):
		for j in range(W):
			x = depth[(i, j)] / 1000.0 / fx * (j - ox)
			y = depth[(i, j)] / 1000.0 / fy * (i - oy)
			z = depth[(i, j)] / 1000.0
			vh = mesh.add_vertex([x, y, z])
			vh_list.append(vh)

	for i in range(H - 1):
		for j in range(W - 1):
			mesh.add_face(vh_list[(i * W + j)], vh_list[((i + 1) * W + j)],
						  vh_list[((i + 1) * W + j + 1)])
			mesh.add_face(vh_list[(i * W + j)], vh_list[((i + 1) * W + j + 1)],
						  vh_list[(i * W + j + 1)])

	assert (~((mask != 0) & (depth == 0))).all
	for i in range(H):
		for j in range(W):
			if mask[(i, j)]==0:
				mesh.delete_vertex(vh_list[(i * W + j)], True)

	mesh.garbage_collection()
	om.write_mesh(save_obj_path, mesh)
	return mesh


def create_cluster(mask, save_cluster_path):
	labeled_img, num = label(mask,
							 connectivity=2,
							 background=0,
							 return_num=True)
	cv2.imwrite(save_cluster_path, labeled_img)


def sort_dist(data):
	i, points, nodes = data
	dist1 = ((points[i:(i + 1), :] - nodes)**2).sum(1)
	dist_arg_sort1 = np.argsort(dist1, 0)
	return dist_arg_sort1[:num_adja]


def fillHole(im_in):
	# cv2.imwrite("test1.png", im_in)
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0, 0), 255)

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
	labeled_img, num = label(bw_img,
							 connectivity=2,
							 background=0,
							 return_num=True)
	max_label = 0
	max_num = 0
	for i in range(1, num + 1):
		if np.sum(labeled_img == i) > max_num:
			max_num = np.sum(labeled_img == i)
			max_label = i
	lcc = (labeled_img == max_label)

	return lcc.astype(np.uint8) * 255


def FilterAndSave(depth, color, parsing, save_color_path, save_depth_path,
				  save_mask_path, save_parsing_path, save_cluster_path):
	list1 = [0, 13, 14, 15, 16, 17, 18]
	backgroud = np.full(depth.shape, False, dtype=bool)
	for i in range(len(list1)):
		backgroud = backgroud | (parsing == list1[i])
	mask = (~backgroud)

	#保留最大连通区域（可选）
	new_mask = largestConnectComponent(mask)
	new_color = cv2.bitwise_and(color, color, mask=new_mask)

	new_mask = ((new_mask!=0) & (depth!=0)).astype(np.uint8) * 255
	new_depth = cv2.bitwise_and(depth, depth, mask=new_mask)
	new_parsing = cv2.bitwise_and(parsing, parsing, mask=new_mask)

	cv2.imwrite(save_color_path, new_color)
	cv2.imwrite(save_depth_path, new_depth)
	cv2.imwrite(save_mask_path, new_mask)
	cv2.imwrite(save_parsing_path, new_parsing)

	if save_cluster_path is not None:
		create_cluster(new_mask != 0, save_cluster_path)

	return new_depth, new_mask, new_color


#从8邻域内找到替补点
def search_new_pose(nodes_mask, u, v):
	if v < (H - 1) and nodes_mask[v + 1, u] == 0: return u, v + 1
	if u < (W - 1) and nodes_mask[v, u + 1] == 0: return u + 1, v
	if u > 1 and nodes_mask[v, u - 1] == 0: return u - 1, v
	if v > 1 and nodes_mask[v - 1, u] == 0: return u, v - 1
	if v < (H - 1) and u < (W - 1) and nodes_mask[v + 1, u + 1] == 0:
		return u + 1, v + 1
	if v > 1 and u > 1 and nodes_mask[v - 1, u - 1] == 0: return u - 1, v - 1
	if v > 1 and u < (W - 1) and nodes_mask[v - 1, u + 1] == 0:
		return u + 1, v - 1
	if u > 1 and v < (H - 1) and nodes_mask[v + 1, u - 1] == 0:
		return u - 1, v + 1
	return u, v


"""
对于每对数据，已有source和target帧的color、depth和parsing label信息，根据已有工具计算mask，cluster(连通区域label)，
deformation graph， 每个像素的邻近节点id

data_list: src_color.jpg, src_depth.png, tar_color.jpg, tar_depth.png
"""


def MainProcess(data_list):
	# try:
	line = data_list
	color_path = line[0]
	depth_path = color_path.replace("color.jpg", "depth.png")

	parsing_path = color_path.replace(raw_data_root,
									  save_data_root).replace(
										  "color.jpg", "parsing_label.png")
	fpath, src_fname = os.path.split(parsing_path)
	parsing_root = fpath + "/parsing_show/"
	parsing_path = parsing_root + src_fname

	save_parsing_root = fpath + "/parsing/"
	if not os.path.exists(save_parsing_root):
		os.makedirs(save_parsing_root)
	save_parsing_path = save_parsing_root + src_fname.replace("parsing_label.png", "parsing.png")


	assert(os.path.exists(parsing_path))

	save_color_root = fpath + "/color/"
	if not os.path.exists(save_color_root):
		os.makedirs(save_color_root)
	save_color_path = save_color_root + src_fname.replace("parsing_label.png", "color.png")

	save_mask_root = fpath + "/mask/"
	if not os.path.exists(save_mask_root):
		os.makedirs(save_mask_root)
	save_mask_path = save_mask_root + src_fname.replace("parsing_label.png", "mask.png")

	save_depth_root = fpath + "/depth/"
	if not os.path.exists(save_depth_root):
		os.makedirs(save_depth_root)
	save_depth_path = save_depth_root + src_fname.replace("parsing_label.png", "depth.png")

	save_cluster_root = fpath + "/cluster/"
	if not os.path.exists(save_cluster_root):
		os.makedirs(save_cluster_root)
	save_cluster_path = save_cluster_root + src_fname.replace("parsing_label.png", "cluster.png")

	save_landmark_68_root = fpath + "/landmark_68/"
	if not os.path.exists(save_landmark_68_root):
		os.makedirs(save_landmark_68_root)
	landmark_68_path = color_path.replace("color.jpg", "info.txt")
	save_landmark_68_path = save_landmark_68_root + src_fname.replace("parsing_label.png", "info.txt")
	if os.path.exists(landmark_68_path) and not os.path.exists(save_landmark_68_path):
		shutil.copy(landmark_68_path, save_landmark_68_path)

	save_landmark_240_root = fpath + "/landmark_240/"
	if not os.path.exists(save_landmark_240_root):
		os.makedirs(save_landmark_240_root)
	landmark_240_path = color_path.replace("color.jpg", "240info.txt")
	save_landmark_240_path = save_landmark_240_root + src_fname.replace("parsing_label.png", "pts.txt")
	if os.path.exists(landmark_240_path) and not os.path.exists(save_landmark_240_path):
		shutil.copy(landmark_240_path, save_landmark_240_path)

	save_graph_root = fpath + "/graph_r12/"
	if not os.path.exists(save_graph_root):
		os.makedirs(save_graph_root)
	save_graph_path = save_graph_root + src_fname.replace("parsing_label.png", "graph.bin")

	save_neighbID_root = fpath + "/neighbID_r12/"
	if not os.path.exists(save_neighbID_root):
		os.makedirs(save_neighbID_root)
	save_neighb_path = save_neighbID_root + src_fname.replace("parsing_label.png", "neighbID.bin")

	save_mesh_graph_root = fpath + "/mesh_graph/"
	if not os.path.exists(save_mesh_graph_root):
		os.makedirs(save_mesh_graph_root)


	#计算target信息
	tar_color_path = line[1]
	tar_depth_path = tar_color_path.replace("color.jpg", "depth.png")


	tar_parsing_path = tar_color_path.replace(raw_data_root,
											save_data_root).replace(
												"color.jpg", "parsing_label.png")
	_, tar_fname = os.path.split(tar_parsing_path)
	
	save_tar_color_path = save_color_root + tar_fname.replace(
		"parsing_label.png", "color.png")
	if not os.path.exists(save_tar_color_path):
		tar_parsing_path = parsing_root + tar_fname
		# logging.info("parsing label of this data is exists: ", os.path.exists(tar_parsing_path))
		save_tar_parsing_root = fpath + "/parsing/"
		if not os.path.exists(save_tar_parsing_root):
			os.makedirs(save_tar_parsing_root)
		save_tar_parsing_path = save_tar_parsing_root + tar_fname.replace("parsing_label.png", "parsing.png")


		save_tar_depth_path = save_depth_root + tar_fname.replace(
			"parsing_label.png", "depth.png")
		save_tar_mask_path = save_mask_root + tar_fname.replace(
			"parsing_label.png", "mask.png")
		save_tar_cluster_path = save_cluster_root + tar_fname.replace(
			"parsing_label.png", "cluster.png")
		save_tar_obj_path = save_mesh_graph_root + tar_fname.replace(
			"parsing_label.png", "mesh.obj")

		tar_landmark_68_path = tar_color_path.replace("color.jpg", "info.txt")
		save_tar_landmark_68_path = save_landmark_68_root + tar_fname.replace("parsing_label.png", "info.txt")
		if os.path.exists(tar_landmark_68_path) and not os.path.exists(save_tar_landmark_68_path):
			shutil.copy(tar_landmark_68_path, save_tar_landmark_68_path)

		tar_landmark_240_path = tar_color_path.replace("color.jpg", "240info.txt")
		save_tar_landmark_240_path = save_landmark_240_root + tar_fname.replace("parsing_label.png", "pts.txt")
		if os.path.exists(tar_landmark_240_path) and not os.path.exists(save_tar_landmark_240_path):
			shutil.copy(tar_landmark_240_path, save_tar_landmark_240_path)

		if not os.path.exists(save_tar_parsing_path) or not os.path.exists(
				save_tar_color_path) or not os.path.exists(
					save_tar_depth_path) or not os.path.exists(
						save_tar_mask_path) or not os.path.exists(
							save_tar_cluster_path) or not os.path.exists(
								save_tar_obj_path):
			tar_color = cv2.imread(tar_color_path)
			tar_depth = cv2.imread(tar_depth_path, -1)
			tar_parsing = cv2.imread(tar_parsing_path, -1)

			tar_depth, tar_mask, tar_color = FilterAndSave(
				tar_depth, tar_color, tar_parsing, save_tar_color_path,
				save_tar_depth_path, save_tar_mask_path, save_tar_parsing_path,
				save_tar_cluster_path)
			create_obj(save_tar_obj_path, tar_depth, tar_mask)

	if os.path.exists(save_graph_path) and os.path.exists(save_neighb_path):
		logging.info("exists")
		return

	color = cv2.imread(color_path)
	depth = cv2.imread(depth_path, -1)
	parsing = cv2.imread(parsing_path, -1)
	#根据parsing信息生成mask，计算连通label, 同时更新color, depth, parsing，
	depth, mask, color = FilterAndSave(depth, color, parsing, save_color_path,
									save_depth_path, save_mask_path,
									save_parsing_path, save_cluster_path)

	#从depth构建mesh
	save_obj_path = save_mesh_graph_root + src_fname.replace(
		"parsing_label.png", "mesh.obj")
	# if not os.path.exists(save_obj_path):
		# create_obj(save_obj_path, depth, mask)
	create_obj(save_obj_path, depth, mask)


	#构建deformation_graph:保存了节点和边的信息
	os.system(
		'./tools/BuildGraph_face/bin/BuildGraph '
		+ save_obj_path + ' ' + save_mesh_graph_root)

	src_mesh = om.read_trimesh(save_obj_path)
	nodes_id = np.loadtxt(
		save_mesh_graph_root +
		src_fname.replace("parsing_label.png", "mesh_nodes.txt")).astype(np.uint32)
	edges = np.loadtxt(
		save_mesh_graph_root +
		src_fname.replace("parsing_label.png", "mesh_edges.txt"))[:, :2].astype(
			np.uint16)
	edges_invert = list(edges[:, [1, 0]])
	edges = list(edges) + edges_invert
	edges_array = np.array(list(set(tuple(edge) for edge in edges)))

	#将graph投影到2d image上
	points = src_mesh.points()
	nodes_mask = np.full((H, W), 0, dtype=np.int32)
	for i in range(nodes_id.shape[0]):
		point = points[nodes_id[i], :]
		xy = [point[0] / point[2], point[1] / point[2]]
		u = int(math.floor(fx * xy[0] + ox))
		v = int(math.floor(fy * xy[1] + oy))
		if nodes_mask[v, u] != 0:
			u, v = search_new_pose(nodes_mask, u, v)
		if nodes_mask[v, u] != 0:
			logging.debug("cover nodes: " + str(nodes_mask[v, u]) + "\n")
		nodes_mask[v, u] = i + 1

	nodes_mask = nodes_mask.reshape(-1)
	nodes_index = np.transpose(np.array(nodes_mask.nonzero()))[:, 0]

	dict1 = {}
	for i in range(nodes_index.shape[0]):
		dict1[str(nodes_mask[nodes_index[i]] - 1)] = i

	edges_array_new = np.zeros_like(edges_array)
	for i in range(edges_array.shape[0]):
		edges_array_new[i, 0] = dict1[str(edges_array[i, 0])]
		edges_array_new[i, 1] = dict1[str(edges_array[i, 1])]

	#save graph
	nodes_index = np.array(nodes_index)
	save_graph(nodes_index, edges_array_new, save_graph_path)

	#计算每个有效像素点邻近节点的id，并保存
	points = np.zeros((H, W, 3))
	points[:, :, 2] = depth / 1000.0
	points[:, :, 0] = (pos_array_x - ox) / fx * depth / 1000.0
	points[:, :, 1] = (pos_array_y - oy) / fy * depth / 1000.0
	points = points.transpose(2, 0, 1).reshape(3, -1).transpose(1, 0)

	nodes = points[nodes_index]
	points_valid = points[(mask != 0).reshape(-1)]

	lists = [[i, points_valid, nodes] for i in range(points_valid.shape[0])]
	pool = ThreadPool()
	neig_ids = pool.map(sort_dist, lists)
	pool.close()
	pool.join()
	neig_nodes_ids = np.array(neig_ids).astype(np.uint16)
	with open(save_neighb_path, 'wb') as f:
		pickle.dump(neig_nodes_ids, f)

	#show graph
	if is_viz:
		show_graph(save_graph_path, save_neighb_path, color, mask)
		exit(0)

def MainProcess_TestData(data_list):
	# try:
	line = data_list
	color_path = line[0]
	depth_path = color_path.replace("color.jpg", "depth.png")

	parsing_path = color_path.replace(raw_data_root,
									  save_data_root).replace(
										  "color.jpg", "parsing_label.png")
	fpath, src_fname = os.path.split(parsing_path)
	parsing_root = fpath + "/parsing_show/"
	parsing_path = parsing_root + src_fname

	save_parsing_root = fpath + "/parsing/"
	if not os.path.exists(save_parsing_root):
		os.makedirs(save_parsing_root)
	save_parsing_path = save_parsing_root + src_fname.replace("parsing_label.png", "parsing.png")
	assert(os.path.exists(parsing_path))

	save_color_root = fpath + "/color/"
	if not os.path.exists(save_color_root):
		os.makedirs(save_color_root)
	save_color_path = save_color_root + src_fname.replace("parsing_label.png", "color.png")

	save_mask_root = fpath + "/mask/"
	if not os.path.exists(save_mask_root):
		os.makedirs(save_mask_root)
	save_mask_path = save_mask_root + src_fname.replace("parsing_label.png", "mask.png")

	save_depth_root = fpath + "/depth/"
	if not os.path.exists(save_depth_root):
		os.makedirs(save_depth_root)
	save_depth_path = save_depth_root + src_fname.replace("parsing_label.png", "depth.png")

	# save_landmark_68_root = fpath + "/landmark_68/"
	# if not os.path.exists(save_landmark_68_root):
	# 	os.makedirs(save_landmark_68_root)
	# landmark_68_path = color_path.replace("color.jpg", "info.txt")
	# save_landmark_68_path = save_landmark_68_root + src_fname.replace("parsing_label.png", "info.txt")
	# if os.path.exists(landmark_68_path) and not os.path.exists(save_landmark_68_path):
	# 	shutil.copy(landmark_68_path, save_landmark_68_path)

	# save_landmark_240_root = fpath + "/landmark_240/"
	# if not os.path.exists(save_landmark_240_root):
	# 	os.makedirs(save_landmark_240_root)
	# landmark_240_path = color_path.replace("color.jpg", "240info.txt")
	# save_landmark_240_path = save_landmark_240_root + src_fname.replace("parsing_label.png", "pts.txt")
	# if os.path.exists(landmark_240_path) and not os.path.exists(save_landmark_240_path):
	# 	shutil.copy(landmark_240_path, save_landmark_240_path)

	color = cv2.imread(color_path)
	depth = cv2.imread(depth_path, -1)
	parsing = cv2.imread(parsing_path, -1)
	#根据parsing信息生成mask，计算连通label, 同时更新color, depth, parsing，
	depth, mask, color = FilterAndSave(depth, color, parsing, save_color_path,
									   save_depth_path, save_mask_path,
									   save_parsing_path, None)



	#计算target信息
	tar_color_path = line[1]
	tar_depth_path = tar_color_path.replace("color.jpg", "depth.png")


	tar_parsing_path = tar_color_path.replace(raw_data_root,
											  save_data_root).replace(
												  "color.jpg", "parsing_label.png")
	fpath, tar_fname = os.path.split(tar_parsing_path)
	tar_parsing_path = parsing_root + tar_fname

	save_tar_parsing_root = fpath + "/parsing/"
	if not os.path.exists(save_tar_parsing_root):
		os.makedirs(save_tar_parsing_root)
	save_tar_parsing_path = save_tar_parsing_root + tar_fname.replace("parsing_label.png", "parsing.png")

	save_tar_color_path = save_color_root + tar_fname.replace(
		"parsing_label.png", "color.png")
	save_tar_depth_path = save_depth_root + tar_fname.replace(
		"parsing_label.png", "depth.png")
	save_tar_mask_path = save_mask_root + tar_fname.replace(
		"parsing_label.png", "mask.png")

	if not os.path.exists(save_tar_parsing_path) or not os.path.exists(
			save_tar_color_path) or not os.path.exists(
				save_tar_depth_path) or not os.path.exists(
					save_tar_mask_path):
		tar_color = cv2.imread(tar_color_path)
		tar_depth = cv2.imread(tar_depth_path, -1)
		tar_parsing = cv2.imread(tar_parsing_path, -1)

		tar_depth, tar_mask, tar_color = FilterAndSave(
			tar_depth, tar_color, tar_parsing, save_tar_color_path,
			save_tar_depth_path, save_tar_mask_path, save_tar_parsing_path,
			None)




if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	raw_data_root = "iphoneX_face/"
	save_data_root = "FaceFlowData/"

	num_adja = 8
	num_target = 1
	is_viz = False
	H = 640
	W = 480
	ox = 240
	oy = 320
	fx = 593.7
	fy = 593.7

	tmp_array = np.ones((1, W), int)
	pos_array_y = np.zeros((1, W), int)
	for i in range(1, H):
		pos_array_y = np.vstack([pos_array_y, i * tmp_array])
	tmp_array = np.ones((H, 1), int)
	pos_array_x = np.zeros([H, 1], int)
	for i in range(1, W):
		pos_array_x = np.hstack([pos_array_x, i * tmp_array])

	frame_path = sys.argv[1]
	frame_txt = open(frame_path, "r")
	lines = frame_txt.readlines()
	logging.info("number of datas: " + str(len(lines)) + "\n")

	phase = sys.argv[2]
	save_data_root = save_data_root + phase + "/"

	if True:
		is_split = int(sys.argv[3])
		i = 0
		if is_split == 1:
			start = int(sys.argv[4])
			end = int(sys.argv[5])
			lines = lines[start:end]
			i = start
		for line in lines:
			try:
				line = line.strip().split()
				if phase == "train" or phase == "val":
					MainProcess(line)
				elif phase == "test":
					MainProcess_TestData(line)
				i += 1
			except Exception as e:
				print(str(e))
				continue
		if is_split == 1:
			logging.info("Process Done!" + str(start) + "_" + str(end) + "\n")
		else:
			logging.info("Process Done!")
