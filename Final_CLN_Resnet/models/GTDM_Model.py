import numpy as np
import torch
import torch.nn as nn
# in-folder debugging
# import adapters as adapters
# import backbones as backbones
# import output_head as output_head
# from vit_dev import ViT, LIMUBertEnc, TransformerDec

import models.adapters as adapters
import models.backbones as backbones
import models.output_head as output_head
from models.vit_dev import ViT, LIMUBertEnc, TransformerDec, TransformerEnc, OrdinaryTransformerEnc
from models.adaptive_conv import AdaptKernel, adaptive_conv
from torchvision import datasets, transforms, models
from torchsummary import summary
from einops import rearrange, repeat
from collections import deque
from models.PoseExpansion import PoseExpand
from scipy.spatial.transform import Rotation as R
from torchvision.models import resnet50, ResNet50_Weights, resnet18
from einops import rearrange, repeat
import random
from models.ConditionalLN import ConditionalLN

# TODO CAN WE TRAIN ON MSE AND FINE TUNE ON NLL NOW THAT THE MEANS ARENT AWFUL??
class GTDM(nn.Module):
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes):
        super(GTDM, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Specify parameters for our ViT blocks
        dim_vit = 128 # TODO Change back 256, 3
        depth_vit = 6
        heads = 8
        dropout = 0.2
        emb_dropout = 0.2

        # Define base ViT

        model_vision = resnet50(pretrained=True)
        ct = 0
        for child in model_vision.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        self.vision = torch.nn.Sequential(*(list(model_vision.children())[:-1]))

        # self.vision = ViT(image_size=(270, 480), patch_size=(15, 30), dim=dim_vit, depth=depth_vit, heads=heads, 
        #                mlp_dim=3*dim_vit, pool = 'cls', channels = 3, dim_head = dim_vit//heads, dropout=dropout, emb_dropout=emb_dropout)
        self.depth = ViT(image_size=(120, 160), patch_size=(12, 16), dim=dim_vit, depth=depth_vit, heads=heads, 
                        mlp_dim=3*dim_vit, pool = 'cls', channels = 1, dim_head = dim_vit//heads, dropout=dropout, emb_dropout=emb_dropout)
        self.mmWave = ViT(image_size=(256, 16), patch_size=(16, 4), dim=dim_vit, depth=depth_vit, heads=heads, 
                        mlp_dim=3*dim_vit, pool = 'cls', channels = 1, dim_head = dim_vit//heads, dropout=dropout, emb_dropout=emb_dropout)

        self.audio = LIMUBertEnc(sig_size=(64,88), dim=dim_vit, depth=depth_vit, heads=heads, 
                        mlp_dim=3*dim_vit, pool = 'cls', channels = 4, dim_head = dim_vit//heads, dropout=dropout, emb_dropout=emb_dropout)
        # Use ModuleDicts to hold the adapters, which are unique for each node and modality
        self.vision_adapter = nn.ModuleDict() # TODO: Maybe 3 adapters for 3 nodes is enough, we can shift the "modality specific feature -> shared feature" task to backbone cls tokens
        self.depth_adapter = nn.ModuleDict()
        self.audio_adapter = nn.ModuleDict()
        self.mmWave_adapter = nn.ModuleDict()
        for node in valid_nodes:
            node = str(node)
            self.vision_adapter[node] = adapters.Adapter(dim_vit, adapter_hidden_dim).to(device)
            self.depth_adapter[node] = adapters.Adapter(dim_vit, adapter_hidden_dim).to(device)
            self.audio_adapter[node] = adapters.Adapter(dim_vit, adapter_hidden_dim).to(device)
            self.mmWave_adapter[node] = adapters.Adapter(dim_vit, adapter_hidden_dim).to(device)

        self.output_head = output_head.OutputHead()
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes

    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data):
        result_dict = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Backbone -> Adapter -> Output Head -> store in dictionary
        if ('zed_camera_left' in self.valid_mods):
            for node in self.valid_nodes:
                node = str(node)
                out = self.vision(data[('zed_camera_left', 'node_' + str(node))])
                out = torch.squeeze(out)
                out = self.vision_adapter[node](out)
                result_dict[('img', 'node_' + str(node))] = self.output_head(out)
        if ('realsense_camera_depth' in self.valid_mods):
            for node in self.valid_nodes:
                node = str(node)
                currData = torch.unsqueeze(data[('realsense_camera_depth', 'node_' + str(node))], 1)
                #currData = currData.repeat(1, 3, 1, 1) # TODO GET RID OF THIS, only for resnet
                out = self.depth(currData)
                out = torch.squeeze(out)
                out = self.depth_adapter[node](out)
                result_dict[('depth', 'node_' + str(node))] = self.output_head(out)
        if ('mic_waveform' in self.valid_mods):
            for node in self.valid_nodes:
                node = str(node)
                audio_sig = data[('mic_waveform', 'node_' + str(node))].cpu().numpy()
                audio_sig_windowed = np.array([np.squeeze(audio_sig[:,:,i:i+88]) for i in range(0,1056-88+1,44)]) # 5ms window x 50% overlap
                if (len(audio_sig_windowed.shape) == 3):
                    audio_sig_windowed = np.expand_dims(audio_sig_windowed, 1)
                audio_sig_windowed = torch.tensor(rearrange(audio_sig_windowed, "w b c l -> b w l c")).to(device) # bs x 64 x 88 x 4 

                out = self.audio(audio_sig_windowed.float())
                out = self.audio_adapter[node](out)
                result_dict[('audio', 'node_' + str(node))] = self.output_head(out)
        if ('range_doppler' in self.valid_mods):
            for node in self.valid_nodes:
                node = str(node)
                out = self.mmWave(torch.unsqueeze(data[('range_doppler', 'node_' + str(node))], 1))
                out = self.mmWave_adapter[node](out)
                result_dict[('mmWave', 'node_' + str(node))] = self.output_head(out) # output_head returns a result of which we take the 'dist' key
        return result_dict
    
num_bns = 0
def replace_layers(model, old, new, target):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new, target)
            
        if isinstance(module, old):
            if (target in n):
                print("Replaced", n, module)
                ## simple module
                setattr(model, n, new)
            

def print_layers(model, curr):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            print_layers(module, curr)
            
        if isinstance(module, curr):
            #print("Found at", n)
            print(n, getattr(model, n))

    
class GTDM_Early(nn.Module):
    def __init__(self, adapter_hidden_dim, valid_mods, valid_nodes):
        super(GTDM_Early, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_vit = 128
        dim_dec = 256
        depth_vit = 6
        depth_dec = 3
        heads = 8
        dropout = 0.2
        emb_dropout = 0.2
        n_kernels = 6
        # CHANGE: Load appropriate weight directory
        All_Weight_Dict = torch.load('./All_Modalities_Pretrain.pt')

        # Define backbones and load the weights into backbones
        model_vision = resnet18(pretrained=True)
        ct = 0
        for child in model_vision.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

        self.vision = torch.nn.Sequential(*(list(model_vision.children())[:-1]))
        replace_layers(self.vision, nn.BatchNorm2d, ConditionalLN((1, 1, 1)), 'bn1')
        replace_layers(self.vision, nn.BatchNorm2d, ConditionalLN((1, 1, 1)), 'bn2')
        print_layers(self.vision, ConditionalLN)

        model_depth = resnet18(pretrained=True)
        ct = 0
        for child in model_depth.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

        self.depth = torch.nn.Sequential(*(list(model_depth.children())[:-1]))
        replace_layers(self.depth, nn.BatchNorm2d, ConditionalLN((1, 1, 1)), 'bn1')
        replace_layers(self.depth, nn.BatchNorm2d, ConditionalLN((1, 1, 1)), 'bn2')
        print_layers(self.depth, ConditionalLN)

        self.mmWave = ViT(image_size=(256, 16), patch_size=(16, 4), dim=dim_vit, depth=depth_vit, heads=heads, 
                        mlp_dim=3*dim_vit, pool = 'cls', channels = 1, dim_head = dim_vit//heads, dropout=dropout, emb_dropout=emb_dropout)
        reduced_state_dict = {k[7:]: v for k, v in All_Weight_Dict.items() if k[7:] in self.mmWave.state_dict() and 'mmWave' in k}
        self.mmWave.load_state_dict(reduced_state_dict)

        self.audio = LIMUBertEnc(sig_size=(64,88), dim=dim_vit, depth=depth_vit, heads=heads, 
                        mlp_dim=3*dim_vit, pool = 'cls', channels = 4, dim_head = dim_vit//heads, dropout=dropout, emb_dropout=emb_dropout)
        reduced_state_dict = {k[6:]: v for k, v in All_Weight_Dict.items() if k[6:] in self.audio.state_dict() and 'audio' in k}
        self.audio.load_state_dict(reduced_state_dict)
        # Use encoder to combine the information, 3 layers to have better crossmodal learning
        self.encoder = OrdinaryTransformerEnc(dim=dim_dec, depth=depth_dec, heads=heads, dim_head=dim_vit//heads, mlp_dim=3*dim_dec, dropout=emb_dropout)
        
        self.vision_adapter = adapters.Adapter(512, adapter_hidden_dim).to(device)
        self.depth_adapter = adapters.Adapter(512, adapter_hidden_dim).to(device)
        self.audio_adapter = adapters.Adapter(dim_vit, adapter_hidden_dim).to(device)
        self.mmWave_adapter = adapters.Adapter(dim_vit, adapter_hidden_dim).to(device)
        self.output_head = output_head.OutputHead()

        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes

        # Num transformer encoders
        self.gammas = np.zeros(2 * depth_vit + 1)
        self.betas = np.zeros(2 * depth_vit + 1)
        self.pose_expand = nn.ModuleDict()
        
        # We want output size 32 for resnet
        self.pose_expand['zed_camera_left'] = PoseExpand(7.5)
        self.pose_expand['realsense_camera_depth'] = PoseExpand(7.5)
        self.pose_expand['range_doppler'] = PoseExpand(depth_vit)
        self.pose_expand['mic_waveform'] = PoseExpand(depth_vit)

                

    # Returns an dictionary of object results with (modality, node) as the keys
    def forward(self, data):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node_pose = []
        node_positions = []
        node_rotations = []
        # Each element in the batch potentially has their own rot and pos
        for i in range(data[('mocap', 'mocap')]['node_rot'].shape[0]):
            node_rot = data[('mocap', 'mocap')]['node_rot'][i]
            node_pos = data[('mocap', 'mocap')]['node_pos'][i]
            node_rotations.append(node_rot)
            node_positions.append(node_pos)
        node_rotations = torch.stack(node_rotations)
        node_positions = torch.stack(node_positions)
        
        batch_size, n_nodes, _ = node_positions.shape
        node_rot_flat = rearrange(node_rotations, "b n t -> (b n) t")
        node_rot_flat = node_rot_flat.cpu()
        node_quat = torch.tensor(np.array([R.from_matrix(each.reshape((3,3))).as_quat() for each in node_rot_flat]))
        node_quat = rearrange(node_quat, "(b n) t -> b n t", b=batch_size)
        #node_quat = node_quat.to(device)
        node_pose_quat = torch.cat((node_quat, node_positions * 0.01), dim=2).float().to(device)
        node_pose = torch.transpose(node_pose_quat, 0, 1) # 3 nodes x 128 batch size x 7 quat + position

        node_pose_quat = rearrange(node_pose_quat, "b n p -> (b n) p")

        result_dict = {}
        node_pose = node_pose.to(device)
        outlist = []

        ran = False

        
        while (ran == False):

            if ('zed_camera_left' in self.valid_mods and (random.uniform(0, 1) > 0.5 or not self.training)):
                curr_mod = 'zed_camera_left'
                ran = True
                for node in self.valid_nodes:

                    curr_node_poses = node_pose[node - 1]
                    ln_params = self.pose_expand[curr_mod](curr_node_poses) # 128 x 7 x 2
                    ln_params = torch.transpose(ln_params, 0, 1)
                 
                    # Num layers * batch_size * 2
                    ConditionalLN.param_queue = deque(ln_params)
                    if (node == 1):
                        print("Node_pose zero element:", curr_node_poses[0])
                        print(ln_params[:, 0])

                    node = str(node)
                    out = torch.squeeze(self.vision(data[('zed_camera_left', 'node_' + str(node))]))
                    if (len(out.shape) == 1):
                        out = torch.unsqueeze(out, dim=0)
                    out = self.vision_adapter(out)
                    outlist.append(out)
            if ('realsense_camera_depth' in self.valid_mods and (random.uniform(0, 1) > 0.5 or not self.training)):
                ran = True
                curr_mod = 'realsense_camera_depth'
                for node in self.valid_nodes:
                    curr_node_poses = node_pose[node - 1]
                    ln_params = self.pose_expand[curr_mod](curr_node_poses) # 128 x 7 x 2
                    ln_params = torch.transpose(ln_params, 0, 1)
                    # Num layers * batch_size * 2
                    ConditionalLN.param_queue = deque(ln_params)
                    node = str(node)
                    depth_data = torch.unsqueeze(data[('realsense_camera_depth', 'node_' + str(node))], 1)
                    depth_data = depth_data.repeat(1, 3, 1, 1)
                    out = torch.squeeze(self.depth(depth_data))
                    if (len(out.shape) == 1):
                        out = torch.unsqueeze(out, dim=0)
                    out = self.depth_adapter(out)
                    outlist.append(out)
            if ('mic_waveform' in self.valid_mods and (random.uniform(0, 1) > 0.5 or not self.training)):
                ran = True
                curr_mod = 'mic_waveform'
                for node in self.valid_nodes:
                    curr_node_poses = node_pose[node - 1]
                    ln_params = self.pose_expand[curr_mod](curr_node_poses) # 128 x 7 x 2
                    ln_params = torch.transpose(ln_params, 0, 1)
                    
                    ConditionalLN.param_queue = deque(ln_params)
                    node = str(node)
                    audio_sig = data[('mic_waveform', 'node_' + str(node))].cpu().numpy()
                    audio_sig_windowed = np.array([np.squeeze(audio_sig[:,:,i:i+88]) for i in range(0,1056-88+1,44)]) # 5ms window x 50% overlap
                    if (len(audio_sig_windowed.shape) == 3):
                        audio_sig_windowed = np.expand_dims(audio_sig_windowed, 1)
                    audio_sig_windowed = torch.tensor(rearrange(audio_sig_windowed, "w b c l -> b w l c")).to(device) 
                    out = self.audio(audio_sig_windowed.float())
                    out = self.audio_adapter(out)
                    outlist.append(out)
            if ('range_doppler' in self.valid_mods and (random.uniform(0, 1) > 0.5 or not self.training)):
                ran = True
                curr_mod = 'range_doppler'
                for node in self.valid_nodes:
                    curr_node_poses = node_pose[node - 1]
                    ln_params = self.pose_expand[curr_mod](curr_node_poses) # 128 x 7 x 2
                    ln_params = torch.transpose(ln_params, 0, 1)
                    # Num layers * batch_size * 2
                    ConditionalLN.param_queue = deque(ln_params)
                    # node = str(node)
                    # out = self.vision(data[('zed_camera_left', 'node_' + str(node))], delta_gamma_queue, delta_beta_queue)
                    # outlist.append(self.vision_adapter(out))
                    node = str(node)
                    # delta_beta_queue = deque(list(self.betas))
                    # delta_gamma_queue = deque(list(self.gammas))
                    out = self.mmWave(torch.unsqueeze(data[('range_doppler', 'node_' + str(node))], 1))
                    out = self.mmWave_adapter(out)
                    outlist.append(out)


        #agg_features = torch.concat(outlist, axis=1) # bs x 300 x 256
        agg_features = torch.stack(outlist, dim=1)
        b, n, _ = agg_features.shape

        

        #agg_features = agg_features + self.final_pos_embedding
        out = self.encoder(agg_features) #bs x total_patches x 256
        out = torch.mean(out, dim=1)
        
        result_dict["early_fusion"] = self.output_head(out) # output_head returns a result of which we take the 'dist' key
        return result_dict
    

if __name__== "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_nodes = [1,2,3]
    valid_mods = ["zed_camera_left", 'range_doppler', 'mic_waveform', 'realsense_camera_depth']
    model = GTDM_Early(adapter_hidden_dim=256, valid_mods=valid_mods, valid_nodes=valid_nodes).to(device)
    data = {('zed_camera_left','node_1'): torch.rand(64,3,270,480).to(device),
            ('zed_camera_left','node_2'): torch.rand(64,3,270,480).to(device),
            ('zed_camera_left','node_3'): torch.rand(64,3,270,480).to(device),
            ('realsense_camera_depth', 'node_1'): torch.rand(64,120,160).to(device),
            ('realsense_camera_depth', 'node_2'): torch.rand(64,120,160).to(device),
            ('realsense_camera_depth', 'node_3'): torch.rand(64,120,160).to(device),
            ('range_doppler', 'node_1'):torch.rand(64,256,16).to(device),
            ('range_doppler', 'node_2'):torch.rand(64,256,16).to(device),
            ('range_doppler', 'node_3'):torch.rand(64,256,16).to(device),
            ('mic_waveform', 'node_1'):torch.rand(64,4,1056).to(device),
            ('mic_waveform', 'node_2'):torch.rand(64,4,1056).to(device),
            ('mic_waveform', 'node_3'):torch.rand(64,4,1056).to(device)}
    res = model(data)
    import ipdb; ipdb.set_trace()
    


