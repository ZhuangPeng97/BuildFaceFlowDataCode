import torch 
import numpy as np
import torch.nn as nn
from openmesh import *
import logging
import cv2
import sys
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class NonLinear_3DMM(nn.Module):
    def __init__(self, point_num, nl_id_dim, nl_exp_dim):
        super(NonLinear_3DMM, self).__init__()
        self.point_num = point_num
        self.geo_fc1 = nn.Linear(nl_id_dim + nl_exp_dim, 2048)
        self.geo_fc2 = nn.Linear(2048, 3*self.point_num)
        self.activate_opt = nn.ReLU()

    def get_geo(self, pca_para):
        feature = self.activate_opt(self.geo_fc1(pca_para))
        return self.geo_fc2(feature)

    def forward_geo(self, id_para, exp_para):
        pca_para = torch.cat((id_para, exp_para), 1)
        geometry = self.get_geo(pca_para).reshape(-1, self.point_num, 3)
        return geometry

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def rot_trans_geo(geometry, rot, trans):
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans.view(-1, 3, 1)
    return rott_geo.permute(0, 2, 1)


def euler_trans_geo(geometry, euler, trans):
    rot = euler2rot(euler)
    return rot_trans_geo(geometry, rot, trans)

def np2mesh(mesh, xnp, path):
    mesh.points()[:] = xnp
    write_mesh(path, mesh, binary=True)


def proj_geo_fun(rott_geo, camera_para):
    fx = camera_para[:, 0]
    fy = camera_para[:, 1]
    cx = camera_para[:, 2]
    cy = camera_para[:, 3]

    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]

    fxX = fx[:, None]*X
    fyY = fy[:, None]*Y

    proj_x = -fxX/Z + cx[:, None]
    proj_y = fyY/Z + cy[:, None]

    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def is_in_triangle(v, T_p1, T_p2, T_p3):
    s31 = sign(v, T_p3, T_p1)
    s23 = sign(v, T_p2, T_p3)
    s12 = sign(v, T_p1, T_p2)

    is_neg = (s12 < 0) or (s23 < 0) or (s31 < 0)
    is_pos = (s12 > 0) or (s23 > 0) or (s31 > 0)

    return not (is_neg and is_pos)

def compute_weights(v, T_p1, T_p2, T_p3):
    area = sign(T_p2, T_p3, T_p1) / 2
    if area == 0:
        return 1.0/3, 1.0/3, 1.0/3
    w1 = sign(T_p2, T_p3, v) / 2 / area
    w2 = sign(T_p3, T_p1, v) / 2 / area
    w3 = sign(T_p1, T_p2, v) / 2 / area
    return w1, w2, w3

def judge_visiblity(mesh, proj_geo):
    visiblity = np.array([True] * proj_geo.shape[0])
    pos = (proj_geo[:, 1] * W + proj_geo[:, 0]).tolist()
    z_dict={}
    for i in range(proj_geo.shape[0]):
        if str(pos[i]) in z_dict:
            if proj_geo[i, 2].item()>z_dict[str(pos[i])][1]:
                visiblity[z_dict[str(pos[i])][0]] = False
                z_dict[str(pos[i])] = (i, proj_geo[i, 2].item())
            else:
                visiblity[i] = False
        else:
            z_dict[str(pos[i])] = (i, proj_geo[i, 2].item())

    for fh in mesh.faces():
        triangle = []
        zs = []
        for vh in mesh.fv(fh):
            idx = vh.idx()
            triangle.append([proj_geo[idx, 0].item(), proj_geo[idx, 1].item()])
            zs.append(proj_geo[idx, 2].item())
        min_x = min(triangle[0][0], triangle[1][0], triangle[2][0])
        max_x = max(triangle[0][0], triangle[1][0], triangle[2][0])
        min_y = min(triangle[0][1], triangle[1][1], triangle[2][1])
        max_y = max(triangle[0][1], triangle[1][1], triangle[2][1])

        pix = []
        for x in range(max(0,min_x-1), min(max_x+1, W)):
            for y in range(max(0, min_y-1), min(max_y+1, H)):
                pix.append([x, y])

        for p in pix:
            cur_pos = p[1]*W+p[0]
            if str(cur_pos) not in z_dict:
                continue
            value = z_dict[str(cur_pos)]
            if not visiblity[value[0]]:
                continue
            if is_in_triangle(p, triangle[0], triangle[1], triangle[2]):
                w1, w2, w3 = compute_weights(p, triangle[0], triangle[1], triangle[2])
                z = w1 * zs[0] + w2 * zs[1] + w3 * zs[2]
                if z>value[1]:
                    visiblity[value[0]] = False
    return visiblity



if __name__ == '__main__':
    decoder_nl3dmm = NonLinear_3DMM(34650, 500, 500)
    decoder_nl3dmm.load_state_dict(torch.load('./tools/ShapeFromX/util/nl3dmm.pth'))
    decoder_nl3dmm = decoder_nl3dmm.cuda()
    mesh = read_trimesh('util/sub_mesh.obj')

    logging.basicConfig(level=logging.DEBUG)
    raw_data_root = "iphoneX_face/"
    save_data_root = "FaceFlowData/"
    phase = sys.argv[2]
    save_data_root = save_data_root + phase + "/"

    H = 640
    W = 480

    frame_path = sys.argv[1]
    frame_txt = open(frame_path, "r")

    datalist = []
    lines = frame_txt.readlines()
    frame_txt.close()
    for line in lines:
        line = line.strip().split()
        datalist.append(line[0])
        datalist.append(line[1])

    datalist = datalist[::-1]
    batch_size = 1000
    for i in range(int((len(datalist)-1)/batch_size)+1):
        print(i)
        cur_batch_size = min(batch_size, len(datalist) - i*batch_size)
        color_paths    = []
        mask_paths     = []
        recon_paths    = []
        id_paras       = []
        exp_paras      = []
        transs         = []
        eulers         = []
        for j in range(cur_batch_size):
            color_path = datalist[i*batch_size+j]
            npz_path = color_path.replace(raw_data_root,
                                            save_data_root).replace(
                                                "color.jpg", "paras.npz")
            fpath, src_fname = os.path.split(npz_path)

            mask_root = fpath + "/mask/"
            mask_path = mask_root + src_fname.replace("paras.npz", "mask.png")

            recon_root = fpath + "/reconstruction/"
            if not os.path.exists(recon_root):
                os.makedirs(recon_root)
            recon_path = recon_root + src_fname.replace("paras.npz", "recon.txt")
            if os.path.exists(recon_path):
                # print("recon_path exists")                
                continue
            if (not os.path.exists(color_path)):
                # print(color_path + " not exists")
                continue
            if (not os.path.exists(mask_path)):
                # print(mask_path + " not exists")
                continue
            npz_root = fpath + "/recon_mesh/"
            npz_path = npz_root + src_fname
            npzfile = np.load(npz_path)

            id_para = torch.as_tensor(npzfile['id'])
            exp_para = torch.as_tensor(npzfile['exp'])
            trans = torch.as_tensor(npzfile['trans'])
            euler = torch.as_tensor(npzfile['euler'])

            assert os.path.exists(mask_path), "please generate background mask before project 3d face"
            mask_paths.append(mask_path)
            color_paths.append(color_path)
            recon_paths.append(recon_path)
            id_paras.append(id_para)
            exp_paras.append(exp_para)
            transs.append(trans)
            eulers.append(euler)

        cur_batch_size = len(recon_paths)
        if cur_batch_size==0:
            continue
        id_para  = torch.stack(id_paras).cuda()
        exp_para = torch.stack(exp_paras).cuda()
        trans    = torch.stack(transs).cuda()
        euler    = torch.stack(eulers).cuda()

        cam_para = torch.tensor((594, 594, 240, 320), dtype=torch.float).unsqueeze(0).repeat(cur_batch_size, 1).cuda()

        geometry = decoder_nl3dmm.forward_geo(id_para, exp_para)
        rott_geo = euler_trans_geo(geometry, euler, trans)
        proj_geo = proj_geo_fun(rott_geo, cam_para)
        proj_geo = torch.round(proj_geo).long().detach().cpu().numpy()
        rott_geo = rott_geo.detach().cpu().numpy()
        in_range = (proj_geo[0, :, 1]>=0) & (proj_geo[0, :, 1]<H) & (proj_geo[0, :, 0]>=0) & (proj_geo[0, :, 0]<W)

        for j in range(cur_batch_size):
            try:
                mesh.points()[:] = rott_geo[j,...]
                # write_mesh("debug/"+os.path.basename(recon_paths[j])[:-4]+".obj", mesh)

                visiblity = judge_visiblity(mesh, proj_geo[j, :, :])
                
                mask = (cv2.imread(mask_paths[j], -1) != 0)
                idi = proj_geo[j, :, 1]
                idj = proj_geo[j, :, 0]
                idi[~in_range] = 0
                idj[~in_range] = 0
                in_mask = mask[(idi, idj)].reshape(-1, 1)

                valid_mask = ((visiblity & in_range).reshape(-1, 1) & in_mask).astype(np.float64)
                out        = np.concatenate((proj_geo[j, :, :2], valid_mask), axis=1)
                np.savetxt(recon_paths[j], out, fmt="%d")

                if 0:
                    obj_file = open("debug/"+os.path.basename(recon_paths[j])[:-4]+"_show_proj.obj", "w")
                    px = np.loadtxt(recon_paths[j])
                    color = cv2.imread(color_paths[j])
                    for i in range(px.shape[0]):
                        if px[i, 2]==1:
                            obj_file.write("v " + str(mesh.points()[i, 0]) + " " + str(mesh.points()[i, 1]) + " " + str(mesh.points()[i, 2]) + "\n")
                            cv2.circle(color, (int(px[i, 0]), int(px[i, 1])), 1, (255, 0, 0))
                    obj_file.close()
                    cv2.imwrite("debug/"+os.path.basename(recon_paths[j])[:-4]+"_show_proj.png", color)
                    exit(0)
            except Exception as e:
                print(str(e))
                continue
        print(i)
# mesh = openmesh.read_trimesh('sub_mesh.obj')
# np2mesh(mesh, rott_geo[0,...].detach().cpu().numpy(), 'test.ply')
