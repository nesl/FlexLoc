import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import cv2

# Change this to modify the actual pickle files to do this as part of cachign rather than dynamic
class PickleDataset(Dataset):
    def __init__(self, file_path, valid_mods, valid_nodes):
        self.data = []
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes
        for file_name in tqdm(sorted(os.listdir(file_path)), desc="Loading pickle files into dataset"):
            curr_pickle = pickle.load(open(file_path + '/' + file_name,  'rb'))
            
            # if (curr_pickle[('mic_waveform', 'node_1')].shape != (4, 1056) or curr_pickle[('mic_waveform', 'node_2')].shape != (4, 1056) or 
            #     curr_pickle[('mic_waveform', 'node_3')].shape != (4, 1056)):
            #     import pdb; pdb.set_trace()
            self.data.append(curr_pickle)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Move this to the preprocess pickle place
 
        # for node in self.valid_nodes:
        #     curr_key = ('mic_waveform', 'node_' + str(node))
        #     if self.data[idx][curr_key].shape != (4, 1056):
        #         self.data[idx][curr_key] = np.append(self.data[idx][curr_key], np.zeros((4, 1056 - self.data[idx][curr_key].shape[1])), axis=1)
           
        return {'data': self.data[idx], 'gt_pos': self.data[idx][('mocap', 'mocap')]['gt_positions']}

