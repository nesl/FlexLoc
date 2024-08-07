import os
import glob
import pickle
import numpy as np
import h5py
import torch
import json
import time
import torchaudio
from tqdm import trange, tqdm
from matplotlib.patches import Ellipse, Rectangle
import copy
import cv2
from viz import *

# def rot2angle(rot, return_rads=True):
#     if rot[4] <= 0:
#         rads = np.arcsin(rot[3]) / (2*np.pi)
#     else:
#         rads = np.arcsin(rot[1]) / (2*np.pi)
#     if not return_rads:
#         rads *= 360
#     return rads

# def rot_matrix(angle):
#     rad = 2*np.pi * (angle/360)
#     R = [np.cos(rad), np.sin(rad),-np.sin(rad), np.cos(rad)]
#     R = np.array(R).reshape(2,2)
#     R = torch.from_numpy(R).float()
#     return R


# def gen_rectange(pos, angle, w, h, color='black'):
#     # angle = rot2angle(rot, return_rads=False)
#     rec = Rectangle(xy=([pos[0]-w/2, pos[1]-h/2]), width=w, height=h, angle=angle, rotation_point='center',
#                         edgecolor=color, fc='None', lw=5)
#     corners = rec.get_corners()

#     x = np.arange(0.5,30,1) / 100.0
#     y = np.arange(0.5,15,1) / 100.0
#     X, Y = np.meshgrid(x,y)
#     grid = np.stack([X,Y])
#     grid = torch.from_numpy(grid).float()
#     grid = grid.permute(1,2,0)
#     grid = grid.reshape(-1,2)
#     R = rot_matrix(angle)
#     grid = torch.mm(grid, R)
#     grid[:,0] += corners[0][0]
#     grid[:,1] += corners[0][1]
#     return rec, grid


def convert2dict(f, keys, fname, valid_mods, valid_nodes):
    data = {}
    for ms in tqdm(keys, desc='loading %s' % fname):
        data[ms] = {}
        for k, v in f[ms].items():
            if k == 'mocap': #mocap or node_N
                data[ms]['mocap'] = v[()]
            else: #is node_N
                if k not in valid_nodes:
                    continue
                data[ms][k] = {}
                for k2, v2 in f[ms][k].items():
                    if k2 not in valid_mods:
                        continue
                    if k2 == 'detected_points':
                        data[ms][k][k2] = v2[()]
                    else:
                        data[ms][k][k2] = v2[:]
    return data


def load_chunk(fname, valid_mods, valid_nodes):
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
        keys = np.array(keys).astype(int)
        keys = list(keys.astype(str))
        data = convert2dict(f, keys, fname, valid_mods, valid_nodes)
    return data

def split_list(lst, n):
    # Calculate the size of each chunk, rounding up
    chunk_size = (len(lst) + n - 1) // n
    # Split list into n-sized chunks
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

class DataCacher(object):
    CLASSES = None
    def __init__(self,
                 hdf5_fnames=[],
                 cache_dir= f'/dev/shm/cache_train/',
                 fps=20,
                 valid_mods=['mocap', 'zed_camera_left', 'zed_camera_depth'],
                 valid_nodes=[1,2,3,4],
                 min_x=-2162.78244, max_x=4157.92774,
                 min_y=-1637.84491, max_y=2930.06133,
                 min_z=0.000000000, max_z=903.616290,
                 normalized_position=False,
                 # max_len=None,
                 truck_w=30/100,
                 truck_h=15/100,
                 fifths=None,
                 include_z=True,
                 **kwargs):
        self.valid_mods = valid_mods
        self.valid_nodes = ['node_%d' % n for n in valid_nodes]
        self.cache_dir = cache_dir
        self.min_x = min_x
        self.max_x = max_x
        self.len_x = 7000
        self.min_y = min_y
        self.max_y = max_y
        self.len_y = 5000
        self.min_z = min_z
        self.max_z = max_z
        self.len_z = 1000
        self.normalized_position = normalized_position
        self.truck_w = truck_w
        self.truck_h = truck_h
        self.include_z = include_z
        self.hdf5_fnames = hdf5_fnames
        self.fps = fps
        self.class2idx = {'truck': 1, 'node': 0}
        self.fifths = fifths
        # self.max_len = max_len


    def preprocess_pickles(self):
        for file_name in tqdm(sorted(os.listdir(self.cache_dir)), desc="Preprocessing pickle files"):
            curr_pickle = pickle.load(open(self.cache_dir + '/' + file_name,  'rb'))
            for keys in curr_pickle.keys():
                if ('realsense_camera_depth' in keys):
                    curr_pickle[keys] = cv2.imdecode(curr_pickle[keys], cv2.IMREAD_GRAYSCALE)
                    curr_pickle[keys] = cv2.normalize(curr_pickle[keys], None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                elif ('zed_camera_left' in keys):
                    curr_pickle[keys] = cv2.cvtColor(cv2.imdecode(curr_pickle[keys], 1), cv2.COLOR_BGR2RGB)
                    curr_pickle[keys] = cv2.normalize(curr_pickle[keys], None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    curr_pickle[keys] = np.transpose(curr_pickle[keys], (2, 0, 1))
                elif ('mic_waveform' in keys):
                    curr_pickle[keys] = curr_pickle[keys][:, 1:5]
                    curr_pickle[keys] = np.transpose(curr_pickle[keys])
                    if curr_pickle[keys].shape != (4, 1056):
                        curr_pickle[keys] = np.append(curr_pickle[keys], np.zeros((4, 1056 - curr_pickle[keys].shape[1])), axis=1)
                elif ('azimuth_static' in keys):
                    curr_pickle[keys] = np.nan_to_num(curr_pickle[keys])
            pickle.dump(curr_pickle, open(self.cache_dir + '/' + file_name,  'wb'))

    def cache(self):
        pastK = []
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=False)
            data = {}
            for fname in self.hdf5_fnames:
                chunk = load_chunk(fname, self.valid_mods, self.valid_nodes)
                for ms, val in chunk.items():
                    if ms in data.keys():
                        for k, v in val.items():
                            if k not in pastK:
                                pastK.append(k)
                            if k in data[ms].keys():
                                data[ms][k].update(v)
                            else:
                                data[ms][k] = v
                    else:
                        data[ms] = val

            buffers = self.fill_buffers(data)
            self.active_keys = sorted(buffers[-1].keys())
            
            count = 0
            for i in range(len(buffers)):
                missing = False
                for key in self.active_keys:
                    if key not in buffers[i].keys():
                        missing = True
                if missing:
                    count += 1
                    continue
                else:
                    break
            buffers = buffers[count:]
            if self.fifths is not None:
                chunks = split_list(buffers, 5)
                chunks = [chunks[i] for i in self.fifths]
                buffers = sum(chunks, [])

            

            # if self.max_len is not None:
                # buffers = buffers[0:self.max_len]
        
            for i in trange(len(buffers)):
                buff = buffers[i]
                fname = '%s/tmp_%09d.pickle' % (self.cache_dir, i)
                with open(fname, 'wb') as f:
                    pickle.dump(buff, f)

            self.preprocess_pickles()
        self.fnames = sorted(glob.glob(f'{self.cache_dir}/*.pickle'))
        with open(self.fnames[-1], 'rb') as f:
            buff = pickle.load(f)
            self.active_keys = sorted(buff.keys())
        # self.fnames = self.fnames[0:self.max_len]
        return self.fnames, self.active_keys
    


    def fill_buffers(self, all_data):
        buffers = []
        buff = {}
        # factor = 100 // self.fps
        factor = 1
        num_frames = 0
        keys = sorted(list(all_data.keys()))
        prev_num_objs = None
        for time in tqdm(keys, desc='filling buffers'):
            save_frame = False
            data = all_data[time]
            for key in data.keys():
                if key == 'mocap':
                    mocap_data = json.loads(data['mocap'])
                    if self.normalized_position:
                        gt_pos = torch.tensor([d['normalized_position'] for d in mocap_data])
                    else:
                        gt_pos = torch.tensor([d['position'] for d in mocap_data])  
                        if not torch.sum(torch.isnan(gt_pos)): # a temp fix of the OptiTrack NaN issue
                            last_known_pos = gt_pos
                        else:
                            for pos_row in range(gt_pos.shape[0]):  
                                for pos_col in range(gt_pos.shape[1]):
                                    if torch.isnan(gt_pos[pos_row, pos_col]):
                                        gt_pos[pos_row, pos_col] = last_known_pos[pos_row, pos_col] 

                        # for UCLA data, swap y and z axis since optitrack z is on the horizontal plane
                        # temp = torch.zeros(gt_pos.shape)
                        # # temp[:,0] = gt_pos[:,0]
                        # # temp[:,1] = gt_pos[:,2]
                        # # temp[:,2] = gt_pos[:,1]
                        # gt_pos = temp
                    gt_rot = torch.tensor([d['rotation'] for d in mocap_data])
                    
                    corners, grids = [], []
                    for k in range(len(gt_rot)):
                        angle = rot2angle(gt_rot[k], return_rads=False)
                        rec, grid = gen_rectange(gt_pos[k], angle, w=self.truck_w, h=self.truck_h)
                        corners.append(rec.get_corners())
                        if self.include_z:
                            z_val = gt_pos[k][-1]
                            z_vals = torch.ones(len(grid), 1) * z_val
                            grid = torch.cat([grid, z_vals], dim=-1)
                        grids.append(grid)

                    grids = torch.stack(grids)

                    corners = np.stack(corners)
                    corners = torch.tensor(corners).float()
                    gt_labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])
                    gt_ids = torch.tensor([d['id'] for d in mocap_data])
                    is_node = gt_labels == 0
                    

                    node_rot = gt_rot[is_node]
                    node_pos = gt_pos[is_node] * 100
                    #node_pos = node_pos[..., 0:2]
                    node_ids = gt_ids[is_node]
                    
                    final_mask = ~is_node 
                    if not self.include_z:
                        gt_pos = gt_pos[..., 0:2]
                    gt_pos = gt_pos[final_mask] * 100
                    gt_grid = grids[final_mask] * 100
                    gt_rot = gt_rot[final_mask] 
                    gt_ids = gt_ids[final_mask] - 3
                    for ii in range(len(gt_ids)):
                        gt_ids[ii] = 0 # assuming only one car in the scene
                   
                        
                    buff[('mocap', 'mocap')] = {
                        'gt_positions': gt_pos,
                        #'gt_labels': gt_labels[final_mask].long(),
                        'gt_ids': gt_ids.long(),
                        'gt_rot': gt_rot,
                        'gt_grids': gt_grid,
                        'node_pos': node_pos,
                        'node_ids': node_ids,
                        'node_rot': node_rot
                    }
                    
                    num_frames += 1
                    save_frame = True

                if 'node' in key:
                    for k, v in data[key].items():
                        if k in self.valid_mods:
                            buff[(k, key)] = v
            
            if save_frame and num_frames % factor == 0:
                new_buff = copy.deepcopy(buff)
                buffers.append(new_buff)
        return buffers
