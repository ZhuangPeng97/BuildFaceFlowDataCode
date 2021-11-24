import os
import shutil
import numpy as np
import struct
import pickle
from scipy.sparse import coo_matrix

def load_flow(filename):
    # Flow is stored row-wise in order [channels, height, width].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    flow = None
    with open(filename, 'rb') as fin:
        width = struct.unpack('I', fin.read(4))[0]
        height = struct.unpack('I', fin.read(4))[0]
        channels = struct.unpack('I', fin.read(4))[0]
        n_elems = height * width * channels

        flow = struct.unpack('f' * n_elems, fin.read(n_elems * 4))
        flow = np.asarray(flow, dtype=np.float32).reshape([channels, height, width])

    return flow
    
def load_graph_info(filename):

    assert os.path.isfile(filename), "File not found: {}".format(filename)

    with open(filename, 'rb') as fin:
        edge_total_size = struct.unpack('I', fin.read(4))[0]
        #n_elems = height * width * channels
        edge = struct.unpack('I' * (int(edge_total_size / 4)), fin.read(edge_total_size))
        edge = np.asarray(edge, dtype=np.int32).reshape(-1, 2).transpose()
        Mv_total_size = struct.unpack('I', fin.read(4))[0]
        M_v = struct.unpack('I' * (int(Mv_total_size / 4)), fin.read(Mv_total_size))
        M_v = np.asarray(M_v, dtype=np.int32).reshape(-1, 1)
        
        fx = struct.unpack('f', fin.read(4))[0]
        fy = struct.unpack('f', fin.read(4))[0]
        ox = struct.unpack('f', fin.read(4))[0]
        oy = struct.unpack('f', fin.read(4))[0]
           
    return edge, M_v, fx, fy, ox, oy

def load_graph_corres_info(filename, H, W, max_edges):

    assert os.path.isfile(filename), "File not found: {}".format(filename)

    with open(filename, 'rb') as fin:
        edge_total_size = struct.unpack('I', fin.read(4))[0]
        #n_elems = height * width * channels
        edges = struct.unpack('I' * (int(edge_total_size / 4)), fin.read(edge_total_size))
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2).transpose()
        Mask_total_size = struct.unpack('I', fin.read(4))[0]
        mask_corres_nodes = struct.unpack('I' * (int(Mask_total_size / 4)), fin.read(Mask_total_size))
        mask_corres_nodes = np.asarray(mask_corres_nodes, dtype=np.uint8).reshape(H, W)
        
        edges_extent = np.zeros((2, max_edges), dtype=np.int64)
        edges_mask = np.zeros((max_edges), dtype=np.bool)
        edges_mask[:edges.shape[1]] = 1
        edges_extent[:, :edges.shape[1]] = edges
        fx = struct.unpack('f', fin.read(4))[0]
        fy = struct.unpack('f', fin.read(4))[0]
        ox = struct.unpack('f', fin.read(4))[0]
        oy = struct.unpack('f', fin.read(4))[0]
           
    return edges_extent, edges_mask, mask_corres_nodes, fx, fy, ox, oy

def load_graph_info_new(filename, num_adja):

    assert os.path.isfile(filename), "File not found: {}".format(filename)
    with open(filename, 'rb') as fin:
        edge_total_size = struct.unpack('I', fin.read(4))[0]
        #n_elems = height * width * channels
        edge = struct.unpack('I' * (int(edge_total_size / 4)), fin.read(edge_total_size))
        edge = np.asarray(edge, dtype=np.int64).reshape(-1, 2).transpose()

        edge_valid_total_size = struct.unpack('I', fin.read(4))[0]
        #n_elems = height * width * channels
        edge_mask = struct.unpack('I' * (int(edge_valid_total_size / 4)), fin.read(edge_valid_total_size))
        edge_mask = np.asarray(edge_mask, dtype=np.int32).reshape(-1, 1)

        Mv_total_size = struct.unpack('I', fin.read(4))[0]
        M_v = struct.unpack('I' * (int(Mv_total_size / 4)), fin.read(Mv_total_size))
        M_v = np.asarray(M_v, dtype=np.int32).reshape(-1, 1)
        '''
        List_total_size = struct.unpack('I', fin.read(4))[0]
        List = struct.unpack('I' * (int(List_total_size / 4)), fin.read(List_total_size))
        List = np.asarray(List, dtype=np.int32).reshape(-1, num_adja)
        '''
        fx = struct.unpack('f', fin.read(4))[0]
        fy = struct.unpack('f', fin.read(4))[0]
        ox = struct.unpack('f', fin.read(4))[0]
        oy = struct.unpack('f', fin.read(4))[0]
    return edge, edge_mask, M_v, fx, fy, ox, oy

def load_adja_id_alpha_info(filename, H, W, num_adja):

    assert os.path.isfile(filename), "File not found: {}".format(filename)
    src_v_id = np.zeros((num_adja, H, W), dtype=np.int32)
    src_v_alpha = np.zeros((num_adja, H, W))
    with open(filename, 'rb') as fin:
        key_i, key_j, value_id, value_alpha = pickle.load(fin)
        for i in range(num_adja):
            src_v_id[i, :, :] = coo_matrix((value_id[:, i], (key_i, key_j)), shape=(H, W)).toarray()
            src_v_alpha[i, :, :] = coo_matrix((value_alpha[:, i], (key_i, key_j)), shape=(H, W)).toarray()
    return src_v_id, src_v_alpha

def save_flow(filename, flow_input):
    flow = np.copy(flow_input)

    # Flow is stored row-wise in order [channels, height, width].
    assert len(flow.shape) == 3
    
    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', flow.shape[2]))
        fout.write(struct.pack('I', flow.shape[1]))
        fout.write(struct.pack('I', flow.shape[0]))
        fout.write(struct.pack('={}f'.format(flow.size), *flow.flatten("C")))
