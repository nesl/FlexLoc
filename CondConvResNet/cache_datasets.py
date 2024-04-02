import argparse
import os
import os.path as osp
from cacher import DataCacher
import configs.data_configs as data_configs
from PickleDataset import PickleDataset
from torch.utils.data import DataLoader
# from models.backbones import VisionBackbone
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import cv2
import pickle



def cache_data():
    cache_train = DataCacher(hdf5_fnames = data_configs.trainset['hdf5_fnames'], cache_dir= data_configs.cache_dir + 'train', 
               valid_mods=data_configs.valid_mods, 
               valid_nodes=data_configs.valid_nodes)
    cache_test = DataCacher(hdf5_fnames = data_configs.testset['hdf5_fnames'], cache_dir=data_configs.cache_dir + 'test', 
               valid_mods=data_configs.valid_mods, 
               valid_nodes=data_configs.valid_nodes)
    cache_val = DataCacher(hdf5_fnames = data_configs.valset['hdf5_fnames'], cache_dir=data_configs.cache_dir + 'val', 
               valid_mods=data_configs.valid_mods, 
               valid_nodes=data_configs.valid_nodes)
    cache_train.cache()
    cache_test.cache()
    cache_val.cache()

# def main():

#     # assert args.out or args.eval or args.format_only or args.show \
#         # or args.show_dir, \
#         # ('Please specify at least one operation (save/eval/format/show the '
#          # 'results / save the results) with the argument "--out", "--eval"'
#          # ', "--format-only", "--show" or "--show-dir"')

#     # if args.eval and args.format_only:
#         # raise ValueError('--eval and --format_only cannot be both specified')

#     # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
#         # raise ValueError('The output file must be a pkl file.')
        
#     #building the datasets runs the cacher
#     #future calls (ie during training) will skip the caching step

#     cache_data()
#     data = PickleDataset('/mnt/nfs/vol2/jason/train')
#     for item in data:
#         import pdb; pdb.set_trace()
#     train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
#     print("Success")
#     #build_dataset(cfg.testset)
    

# if __name__ == '__main__':
#     main()
