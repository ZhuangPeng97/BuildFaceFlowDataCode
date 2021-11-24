import os
import openmesh as om
import cv2
import numpy as np
import concurrent.futures
import math
import struct
import logging
import sys

def argmax(iterable):
    return max((enumerate(iterable)), key=(lambda x: x[1]))[0]


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def is_in_triangle(v, T_p1, T_p2, T_p3):
    s31 = sign(v, T_p3, T_p1)
    s23 = sign(v, T_p2, T_p3)
    s12 = sign(v, T_p1, T_p2)
    is_neg = s12 < 0 or s23 < 0 or s31 < 0
    is_pos = s12 > 0 or s23 > 0 or s31 > 0
    return not (is_neg and is_pos)


def compute_weights(v, T_p1, T_p2, T_p3):
    area = math.fabs(sign(T_p2, T_p3, T_p1)) / 2
    if area == 0:
        return (0.0, 0.0, 0.0)
    else:
        w1 = math.fabs(sign(T_p2, T_p3, v)) / 2 / area
        w2 = math.fabs(sign(T_p1, T_p3, v)) / 2 / area
        w3 = math.fabs(sign(T_p1, T_p2, v)) / 2 / area
        return (w1, w2, w3)


def multi(num, arr):
    return [i * num for i in arr]


def judge_visiblity(res3, res2, fset):
    res2 = np.array(res2)
    visiblity = np.array([True] * res2.shape[0])
    pos = (res2[:, 1] * W + res2[:, 0]).tolist()
    z_dict = {}
    for i in range(res2.shape[0]):
        if str(pos[i]) in z_dict:
            if res3[i][2] > z_dict[str(pos[i])][1]:
                visiblity[z_dict[str(pos[i])][0]] = False
                z_dict[str(pos[i])] = (i, res3[i][2])
            else:
                visiblity[i] = False
        else:
            z_dict[str(pos[i])] = (
             i, res3[i][2])

    for fh in fset:
        triangle = []
        zs = []
        for idx in fh:
            triangle.append([res2[(idx - 1, 0)], res2[(idx - 1, 1)]])
            zs.append(res3[(idx - 1)][2])

        min_x = int(min(triangle[0][0], triangle[1][0], triangle[2][0]))
        max_x = int(max(triangle[0][0], triangle[1][0], triangle[2][0]))
        min_y = int(min(triangle[0][1], triangle[1][1], triangle[2][1]))
        max_y = int(max(triangle[0][1], triangle[1][1], triangle[2][1]))
        pix = []
        for x in range(max(0, min_x - 1), min(max_x + 1, W)):
            for y in range(max(0, min_y - 1), min(max_y + 1, H)):
                pix.append([x, y])

        for p in pix:
            cur_pos = p[1] * W + p[0]
            if str(cur_pos) not in z_dict:
                continue
            else:
                value = z_dict[str(cur_pos)]
                if not visiblity[value[0]]:
                    continue
            if is_in_triangle(p, triangle[0], triangle[1], triangle[2]):
                w1, w2, w3 = compute_weights(p, triangle[0], triangle[1], triangle[2])
                z = w1 * zs[0] + w2 * zs[1] + w3 * zs[2]
                if z > value[1]:
                    visiblity[value[0]] = False

    return visiblity


def save_flow(filename, flow_input):
    flow = flow_input
    assert len(flow.shape) == 3
    with open(filename, 'wb') as (fout):
        fout.write(struct.pack('I', flow.shape[2]))
        fout.write(struct.pack('I', flow.shape[1]))
        fout.write(struct.pack('I', flow.shape[0]))
        fout.write((struct.pack)('={}f'.format(flow.size), *flow.flatten('C')))

def bia2d3d(src2, res2, path2, src3, res3, path3, fset):
    if len(src2) != len(res2) :
        print("obj 2d length wrong!!", len(src2), "and", len(res2))
        return
    if len(src3) != len(res3) :
        print("obj 3d length wrong!!", len(src3), "and", len(res3))
        return

    visibility = judge_visiblity(res3, res2, fset)
    
    res2d = np.full((2, H, W), -np.inf)
    res3d = np.full((3, H, W), -np.inf)

    for i in range(len(fset)):
        id1 = fset[i][0]-1
        id2 = fset[i][1]-1
        id3 = fset[i][2]-1

        if (not visibility[id1]) or (not visibility[id2]) or (not visibility[id3]): continue

        p1 = src2[id1]
        p2 = src2[id2]
        p3 = src2[id3]
        res2d_1 = list(map(lambda x: x[0]-x[1], zip(res2[id1], p1)))
        res2d_2 = list(map(lambda x: x[0]-x[1], zip(res2[id2], p2)))
        res2d_3 = list(map(lambda x: x[0]-x[1], zip(res2[id3], p3)))
        res3d_1 = list(map(lambda x: x[0]-x[1], zip(res3[id1], src3[id1])))
        res3d_2 = list(map(lambda x: x[0]-x[1], zip(res3[id2], src3[id2])))
        res3d_3 = list(map(lambda x: x[0]-x[1], zip(res3[id3], src3[id3])))

        triangle = [p1, p2, p3]
        min_x = int(math.floor(min(p1[0], p2[0], p3[0])))
        max_x = int(math.ceil(max(p1[0], p2[0], p3[0])))
        min_y = int(math.floor(min(p1[1], p2[1], p3[1])))
        max_y = int(math.ceil(max(p1[1], p2[1], p3[1])))
        pix = []
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                pix.append([x, y])
        for p in pix:
            if is_in_triangle(p, triangle[0], triangle[1], triangle[2]):
                w1, w2, w3 = compute_weights(p, triangle[0], triangle[1], triangle[2])
                assert(((w1+w2+w3)<=1.000001 and (w1+w2+w3)>=0.999999) or (w1==0 and w2==0 and w3==0))
            
                res2d_average0 = w1 * res2d_1[0] + w2 * res2d_2[0] + w3 * res2d_3[0]
                res2d_average1 = w1 * res2d_1[1] + w2 * res2d_2[1] + w3 * res2d_3[1]
                res3d_average0 = w1 * res3d_1[0] + w2 * res3d_2[0] + w3 * res3d_3[0]
                res3d_average1 = w1 * res3d_1[1] + w2 * res3d_2[1] + w3 * res3d_3[1]
                res3d_average2 = w1 * res3d_1[2] + w2 * res3d_2[2] + w3 * res3d_3[2]

                res2d[0, p[1], p[0]] = res2d_average0
                res2d[1, p[1], p[0]] = res2d_average1

                res3d[0, p[1], p[0]] = res3d_average0
                res3d[1, p[1], p[0]] = res3d_average1
                res3d[2, p[1], p[0]] = res3d_average2

        u_floor_1 = int(math.floor(p1[0]))
        v_floor_1 = int(math.floor(p1[1]))
        if p1[0]>=0 and p1[0]<=(W-1) and p1[1]>=0 and p1[1]<=(H-1):
            if res2d[0, v_floor_1, u_floor_1] == -np.inf:
                res2d[0, v_floor_1, u_floor_1] = res2d_1[0]
                res2d[1, v_floor_1, u_floor_1] = res2d_1[1]
                res3d[0, v_floor_1, u_floor_1] = res3d_1[0]
                res3d[1, v_floor_1, u_floor_1] = res3d_1[1]
                res3d[2, v_floor_1, u_floor_1] = res3d_1[2]	
            if res2d[0, v_floor_1+1, u_floor_1] == -np.inf:
                res2d[0, v_floor_1+1, u_floor_1] = res2d_1[0]
                res2d[1, v_floor_1+1, u_floor_1] = res2d_1[1]
                res3d[0, v_floor_1+1, u_floor_1] = res3d_1[0]
                res3d[1, v_floor_1+1, u_floor_1] = res3d_1[1]
                res3d[2, v_floor_1+1, u_floor_1] = res3d_1[2]					
            if res2d[0, v_floor_1, u_floor_1+1] == -np.inf:
                res2d[0, v_floor_1, u_floor_1+1] = res2d_1[0]
                res2d[1, v_floor_1, u_floor_1+1] = res2d_1[1]
                res3d[0, v_floor_1, u_floor_1+1] = res3d_1[0]
                res3d[1, v_floor_1, u_floor_1+1] = res3d_1[1]
                res3d[2, v_floor_1, u_floor_1+1] = res3d_1[2]
            if res2d[0, v_floor_1+1, u_floor_1+1] == -np.inf:
                res2d[0, v_floor_1+1, u_floor_1+1] = res2d_1[0]
                res2d[1, v_floor_1+1, u_floor_1+1] = res2d_1[1]
                res3d[0, v_floor_1+1, u_floor_1+1] = res3d_1[0]
                res3d[1, v_floor_1+1, u_floor_1+1] = res3d_1[1]
                res3d[2, v_floor_1+1, u_floor_1+1] = res3d_1[2]
        u_floor_1 = int(math.floor(p2[0]))
        v_floor_1 = int(math.floor(p2[1]))
        if p2[0]>=0 and p2[0]<=(W-1) and p2[1]>=0 and p2[1]<=(H-1):
            if res2d[0, v_floor_1, u_floor_1] == -np.inf:
                res2d[0, v_floor_1, u_floor_1] = res2d_2[0]
                res2d[1, v_floor_1, u_floor_1] = res2d_2[1]
                res3d[0, v_floor_1, u_floor_1] = res3d_2[0]
                res3d[1, v_floor_1, u_floor_1] = res3d_2[1]
                res3d[2, v_floor_1, u_floor_1] = res3d_2[2]	
            if res2d[0, v_floor_1+1, u_floor_1] == -np.inf:
                res2d[0, v_floor_1+1, u_floor_1] = res2d_2[0]
                res2d[1, v_floor_1+1, u_floor_1] = res2d_2[1]
                res3d[0, v_floor_1+1, u_floor_1] = res3d_2[0]
                res3d[1, v_floor_1+1, u_floor_1] = res3d_2[1]
                res3d[2, v_floor_1+1, u_floor_1] = res3d_2[2]					
            if res2d[0, v_floor_1, u_floor_1+1] == -np.inf:
                res2d[0, v_floor_1, u_floor_1+1] = res2d_2[0]
                res2d[1, v_floor_1, u_floor_1+1] = res2d_2[1]
                res3d[0, v_floor_1, u_floor_1+1] = res3d_2[0]
                res3d[1, v_floor_1, u_floor_1+1] = res3d_2[1]
                res3d[2, v_floor_1, u_floor_1+1] = res3d_2[2]
            if res2d[0, v_floor_1+1, u_floor_1+1] == -np.inf:
                res2d[0, v_floor_1+1, u_floor_1+1] = res2d_2[0]
                res2d[1, v_floor_1+1, u_floor_1+1] = res2d_2[1]
                res3d[0, v_floor_1+1, u_floor_1+1] = res3d_2[0]
                res3d[1, v_floor_1+1, u_floor_1+1] = res3d_2[1]
                res3d[2, v_floor_1+1, u_floor_1+1] = res3d_2[2]

        u_floor_1 = int(math.floor(p3[0]))
        v_floor_1 = int(math.floor(p3[1]))
        if p3[0]>=0 and p3[0]<=(W-1) and p3[1]>=0 and p3[1]<=(H-1):
            if res2d[0, v_floor_1, u_floor_1] == -np.inf:
                res2d[0, v_floor_1, u_floor_1] = res2d_3[0]
                res2d[1, v_floor_1, u_floor_1] = res2d_3[1]
                res3d[0, v_floor_1, u_floor_1] = res3d_3[0]
                res3d[1, v_floor_1, u_floor_1] = res3d_3[1]
                res3d[2, v_floor_1, u_floor_1] = res3d_3[2]	
            if res2d[0, v_floor_1+1, u_floor_1] == -np.inf:
                res2d[0, v_floor_1+1, u_floor_1] = res2d_3[0]
                res2d[1, v_floor_1+1, u_floor_1] = res2d_3[1]
                res3d[0, v_floor_1+1, u_floor_1] = res3d_3[0]
                res3d[1, v_floor_1+1, u_floor_1] = res3d_3[1]
                res3d[2, v_floor_1+1, u_floor_1] = res3d_3[2]					
            if res2d[0, v_floor_1, u_floor_1+1] == -np.inf:
                res2d[0, v_floor_1, u_floor_1+1] = res2d_3[0]
                res2d[1, v_floor_1, u_floor_1+1] = res2d_3[1]
                res3d[0, v_floor_1, u_floor_1+1] = res3d_3[0]
                res3d[1, v_floor_1, u_floor_1+1] = res3d_3[1]
                res3d[2, v_floor_1, u_floor_1+1] = res3d_3[2]
            if res2d[0, v_floor_1+1, u_floor_1+1] == -np.inf:
                res2d[0, v_floor_1+1, u_floor_1+1] = res2d_3[0]
                res2d[1, v_floor_1+1, u_floor_1+1] = res2d_3[1]
                res3d[0, v_floor_1+1, u_floor_1+1] = res3d_3[0]
                res3d[1, v_floor_1+1, u_floor_1+1] = res3d_3[1]
                res3d[2, v_floor_1+1, u_floor_1+1] = res3d_3[2]
    save_flow(path2, res2d)
    save_flow(path3, res3d)


def get_obj_point(cur_obj):
    """
    return 3d point[x, y, z] and 2d point[u, v]
    vset_3d: 3d point, n * 3
    vest_2d: 2d point, n * 2
    """
    vset_3d = []
    fset = []
    vset_2d = []
    obj_lines = open(cur_obj)
    for line in obj_lines:
        para = line.strip().split(' ')
        if para[0] != 'v':
            if para[0] != 'f':
                continue
        if para[0] == 'v':
            Point_3d = []
            Point_2d = []
            x = float(para[1])
            y = float(para[2])
            z = float(para[3])
            Point_3d.append(x)
            Point_3d.append(y)
            Point_3d.append(z)

            u = int(round(fx * x / z + ox))
            v = int(round(fy * y / z + oy))

            Point_2d.append(u)
            Point_2d.append(v)
            vset_2d.append(Point_2d)
            vset_3d.append(Point_3d)
        else:
            if para[0] == 'f':
                Face = []
                Face.append(int(para[1]))
                Face.append(int(para[2]))
                Face.append(int(para[3]))
                fset.append(Face)

    return (vset_3d, vset_2d, fset)


def MakeFlow(src_id, tar_id, save_obj_root, save_res_root, save_2d_root, save_3d_root):
    path_2d = save_2d_root + src_id + '_' + tar_id + '.oflow'
    path_3d = save_3d_root + src_id + '_' + tar_id + '.sflow'
    if os.path.exists(path_2d) and os.path.exists(path_3d):
        return

    src = save_obj_root + src_id + '_mesh.obj'
    res = save_res_root + src_id + '_' + tar_id + '.obj'


    if not os.path.exists(src) or not os.path.exists(res):
        print(src + " or " + res + " not exists")
        return
    src_3d, src_2d, fset = get_obj_point(src)
    res_3d, res_2d, _ = get_obj_point(res)
    bia2d3d(src_2d, res_2d, path_2d, src_3d, res_3d, path_3d, fset)

def create_obj(save_obj_path, depth, mask, landmark):
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

    land_3d = []
    landmark = np.around(landmark).astype(np.int32)
    point = np.zeros((1, 3))
    points = mesh.points()
    for i in range(landmark.shape[0]):
        ii = landmark[i, 0]
        jj = landmark[i, 1]
        if (ii<H) and (ii>=0) and (jj<W) and (jj>=0):
            point[0, 0] = depth[(ii, jj)] / 1000.0 / fx * (jj - ox)
            point[0, 1] = depth[(ii, jj)] / 1000.0 / fy * (ii - ox)
            point[0, 2] = depth[(ii, jj)] / 1000.0
            dist = ((point - points) ** 2).sum(1)
            id_ = np.argmin(dist)
            land_3d.append([id_, 1])
        else:
            land_3d.append([0, 0])

    om.write_mesh(save_obj_path, mesh)
    return mesh, np.array(land_3d)

def registration(save_obj_root, save_res_root, landmark_root, src_landmark_path, tar_landmark_path, src_id, tar_id):
    """
    registration source to target;
    """
    res_name = src_id + '_' + tar_id
    save_src_obj_path = save_obj_root + src_id + "_mesh.obj"
    save_tar_obj_path = save_obj_root + tar_id + "_mesh.obj"
    if os.path.exists(save_res_root + res_name + '.obj') and os.path.exists(save_src_obj_path) and os.path.exists(save_tar_obj_path):
        return
    save_land_path = landmark_root + src_id + '_' + tar_id + '_3d.txt'
    assert(os.path.exists(src_landmark_path) and os.path.exists(tar_landmark_path))



    if not os.path.exists(save_src_obj_path) or not os.path.exists(save_tar_obj_path) or not os.path.exists(save_land_path):
        src_depth = cv2.imread(save_obj_root.replace('mesh', 'depth') + "/" + src_id + "_depth.png", -1)
        src_mask = cv2.imread(save_obj_root.replace('mesh', 'mask') + "/" + src_id + "_mask.png", -1)
        src_landmark = np.loadtxt(src_landmark_path).reshape(-1, 2)[:, ::-1]
        print(0)
        _, src_land_3d = create_obj(save_src_obj_path, src_depth, src_mask, src_landmark)
        print(1)
        tar_depth = cv2.imread(save_obj_root.replace('mesh', 'depth') + "/" + tar_id + "_depth.png", -1)
        tar_mask = cv2.imread(save_obj_root.replace('mesh', 'mask') + "/" + tar_id + "_mask.png", -1)
        tar_landmark = np.loadtxt(tar_landmark_path).reshape(-1, 2)[:, ::-1]
        _, tar_land_3d = create_obj(save_tar_obj_path, tar_depth, tar_mask, tar_landmark)
        print(2)

        valid_land = (src_land_3d[:, 1]==1) & (tar_land_3d[:, 1]==1)
        src_land_3d = src_land_3d[:, 0][valid_land]
        tar_land_3d = tar_land_3d[:, 0][valid_land]
        land_3d = np.stack([src_land_3d, tar_land_3d], axis=1)
        np.savetxt(save_land_path, land_3d)
    try:
        os.system('./tools/Fast_RNRR-master/bin/Fast_RNRR ' + save_src_obj_path + ' ' + save_tar_obj_path + ' ' + save_res_root + ' ' + res_name + ' ' + save_land_path)
    except Exception as e:
        print('==========Exception:', e)
        print(save_src_obj_path + ' and ' + save_tar_obj_path + ' skipped!!!')
        return

def build_flow(line):
    src_color_path = line[0]
    tar_color_path = line[1]

    src_landmark_path = src_color_path.replace(raw_data_root,
                                        save_data_root).replace(
                                            "color.jpg", "pts.txt")
    fpath, src_fname = os.path.split(src_landmark_path)
    landmark_root = fpath + "/landmarks/"
    src_landmark_path = landmark_root + src_fname

    tar_landmark_path = tar_color_path.replace(raw_data_root,
                                        save_data_root).replace(
                                            "color.jpg", "pts.txt")
    _, tar_fname = os.path.split(tar_landmark_path)
    tar_landmark_path = landmark_root + tar_fname

    save_obj_root = fpath + "/mesh/"
    save_res_root = fpath + "/registration/"
    save_2d_root = fpath + "/optical_flow/"
    save_3d_root = fpath + "/scene_flow/"

    if not os.path.exists(save_obj_root):
        os.makedirs(save_obj_root)
    if not os.path.exists(save_res_root):
        os.makedirs(save_res_root)
    if not os.path.exists(save_2d_root):
        os.makedirs(save_2d_root)
    if not os.path.exists(save_3d_root):
        os.makedirs(save_3d_root)

    src_id = src_fname.split("_")[0]
    tar_id = tar_fname.split("_")[0]

    registration(save_obj_root, save_res_root, landmark_root, src_landmark_path, tar_landmark_path, src_id, tar_id)

    MakeFlow(src_id, tar_id, save_obj_root, save_res_root, save_2d_root, save_3d_root)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    raw_data_root = "iphoneX_face/"
    save_data_root = "FaceFlowData/"
    phase = sys.argv[2]
    save_data_root = save_data_root + phase + "/"

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
                build_flow(line)
            except Exception as e:
                print(str(e))
                continue
        if is_split == 1:
            logging.info("Process Done!" + str(start) + "_" + str(end) + "\n")
        else:
            logging.info("Process Done!")