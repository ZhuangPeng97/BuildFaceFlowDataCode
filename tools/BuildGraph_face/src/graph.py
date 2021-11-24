import os
import numpy as np
import struct
import math
import csv
import concurrent.futures
import trimesh

def Graph(line):
	line = line.split("_")
	path_obj = "/disk2/zhuang/reg_data/data/" + line[0] + "/" + line[1] + "/" + line[2] + "/"+ line[3] + ".obj"
	save_root = "/disk2/zhuang/reg_data/data/" + line[0] + "/" + line[1] + "/graph/"
	try:
		os.system('./build/BUILDGRAPH '+path_obj+' '+save_root)
		nodes_path = save_root + line[3] + "_init_nodes.txt"
		edges_path = save_root + line[3] + "_init_edges.txt"
		save_nodes_path = save_root + line[3] + "_nodes.txt"
		save_edges_path = save_root + line[3] + "_edges.txt"
		src_mask_path = "/disk2/zhuang/reg_data/data/" + line[0] + "/" + line[1] + "/mask/" + line[3] + "_mask.png"
		if not os.path.exists(data_root + people + "/" + status + "/graph_corres_info/"):
			os.mkdir(data_root + people + "/" + status + "/graph_corres_info/")
		save_bin_path = data_root + people + "/" + status + "/graph_corres_info/"+line[3]+".bin"
		ox = 320.309
		oy = 337.533
		fx = 504.515
		fy = 504.396
		H = 576
		W = 640
		src_mesh = trimesh.load_mesh(path_obj, process=False)
		nodes_old = np.loadtxt(nodes_path).astype(np.uint32)
		edges_old = np.loadtxt(edges_path).astype(np.uint32)
		nodes_pix = np.zeros((nodes_old.shape), dtype=np.uint32)
		for i in range(nodes_old.shape[0]):
			point = src_mesh.vertices[nodes_old[i]]
			u = int(round(fx * point[0] / (point[2] + 1e-8) + ox))
			v = int(round(fy * point[1] / (point[2] + 1e-8) + oy))
			id_ = v * W + u
			nodes_pix[i] = id_
		arg_sort = np.argsort(nodes_pix)
		dict = {}
		nodes = nodes_old[arg_sort]
		for i in range(nodes.shape[0]):
			dict[str(nodes[i])] = i
		edges = np.zeros((edges_old.shape), dtype=np.uint32)
		for i in range(edges_old.shape[0]):
			edges[i, 0] = dict[str(edges_old[i, 0])]
			edges[i, 1] = dict[str(edges_old[i, 1])]
		np.savetxt(save_nodes_path, nodes, fmt='%d')
		np.savetxt(save_edges_path, edges, fmt='%d')

		src_mask = cv2.imread(src_mask_path, -1)
		mask_corres_nodes = np.zeros((H,W),dtype=np.uint8)
		num_valid = (src_mask!=0).sum()
		if num_valid<10000:
			return
		m_non_zero = src_mask.nonzero()
		sample = random.sample(np.arange(num_valid).tolist(), 10000)
		mask_corres_nodes[(m_non_zero[0][sample], m_non_zero[1][sample])] = 1
		assert(mask_corres_nodes.sum()==10000)

		points = src_mesh.vertices
		ids_flag = np.full((H, W), False, np.bool)
		for j in range(nodes.shape[0]):
			point = points[nodes[j], :]
			u = int(round(fx * point[0] / (point[2] + 1e-8) + ox))
			v = int(round(fy * point[1] / (point[2] + 1e-8) + oy))
			while ids_flag[v, u]:
				u += 1
			mask_corres_nodes[v, u] += 2
			ids_flag[v, u] = True
		n = ((mask_corres_nodes==2) + (mask_corres_nodes==3)).sum()
		assert(n == nodes.shape[0])

		with open(save_bin_path, 'wb') as fout:
			edge_total_size = 4*2*edges.shape[0]
			fout.write(struct.pack('I', edge_total_size))
			fout.write(struct.pack('={}I'.format(edges.size), *edges.flatten("C")))
			mask_size = 4*H*W
			fout.write(struct.pack('I', mask_size))
			fout.write(struct.pack('={}I'.format(mask_corres_nodes.size), *mask_corres_nodes.flatten("C")))
			fout.write(struct.pack('f', fx))
			fout.write(struct.pack('f', fy))
			fout.write(struct.pack('f', ox))
			fout.write(struct.pack('f', oy))

	except Exception as e:
		print("==========Exception:", e)
		print(path_obj + " skipped!!!")

if __name__ == '__main__':
	csv_path = "/disk2/zhuang/reg_data/save_data/csv_file/name_selected.csv"
	with open(csv_path, 'r') as f:
		data = csv.reader(f, delimiter=" ")
		lines = [os.path.basename(row[0]) for row in data]
	with concurrent.futures.ProcessPoolExecutor() as executor:
		executor.map(Graph, lines)
