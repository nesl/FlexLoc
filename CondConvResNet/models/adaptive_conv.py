import torch
from torch import nn
from einops import rearrange, repeat
import numpy as np
from scipy.spatial.transform import Rotation as R

class AdaptKernel(nn.Module):
    def __init__(self, num_channels=12, num_kernels=6, kernel_size=5):
        super().__init__()
        self.out_dim = kernel_size * num_channels * num_kernels
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.intermediate_net = nn.Sequential(
            nn.Linear(7, 64), # quarternion for rotation + vector for translation
            nn.GELU(),
            nn.Linear(64, 128)
        )
        self.kernel_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, self.num_channels*num_kernels*kernel_size)
        )
        self.bias_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, num_kernels)
        )
        
    def forward(self, flattened_pose):
        intermediate = self.intermediate_net(flattened_pose)
        conv_kernels = self.kernel_net(intermediate)
        conv_kernels = conv_kernels.reshape(-1, self.num_kernels, self.num_channels, self.kernel_size)
        biases = self.bias_net(intermediate)
        return conv_kernels, biases
        
 
# class HyperConv1D(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, conv_kernels, biases): # x: [bs, 12, 256]
#         bs, n_channel, length = x.shape
#         _, n_kernel, _, k_size = conv_kernels.shape
#         assert n_channel==conv_kernels.shape[2]
#         windowed_x = torch.stack([x[...,i:i+k_size] for i in range(length-k_size+1)])
#         windowed_x = repeat(windowed_x, "n b c k -> n b w c k", w=n_kernel)
#         out = torch.mul(windowed_x, conv_kernels)
#         out = torch.sum(torch.sum(out, axis = 4), axis=3)
#         out = out + biases
#         out = rearrange(out, "n b c -> b c n")
#         return out
    
def adaptive_conv(x, conv_kernels, biases): # x: [bs, 256]
    bs, _ = x.shape
    if len(conv_kernels.shape)==2:
        conv_kernels = rearrange(conv_kernels, "k l -> 1 k l")
    _, n_kernel, k_size = conv_kernels.shape
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    assert bs==conv_kernels.shape[0]
    # same padding
    x = torch.cat((torch.zeros((bs,(k_size-1)//2)).to(device), x, torch.zeros((bs,(k_size-1)//2)).to(device)), axis = 1)
    bs, length = x.shape
    windowed_x = torch.stack([x[...,i:i+k_size] for i in range(length-k_size+1)])
    windowed_x = repeat(windowed_x, "n b k -> n b w k", w=n_kernel)
    out = torch.mul(windowed_x, conv_kernels)
    out = torch.sum(out, axis = 3)
    out = out + biases
    out = rearrange(out, "n b c -> b c n")
    return out


if __name__ == "__main__":
    print("Hi!")
    node_rot = R.from_matrix(np.array([[-0.746781, 0.051433, 0.663078],
                            [-0.216564, -0.961473, -0.169323],
                            [0.628823, -0.270046, 0.729148]]))
    quat = node_rot.as_quat()
    node_loc = np.array([-1.129100, 0.750066, -1.112714])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cam_pos = torch.tensor(np.concatenate((quat,node_loc), dtype=np.float32)).to(device)
    cam_pos = torch.stack((cam_pos,cam_pos,cam_pos,cam_pos,cam_pos,cam_pos,cam_pos))
    features = torch.rand((7,12,256)).to(device)
    hypernet = AdaptKernel(num_channels=12, num_kernels=6, kernel_size=5)
    hypernet.to(device)
    kernels, biases = hypernet(cam_pos)
    # adaptive_conv = HyperConv1D()
    output = adaptive_conv(features, kernels, biases)
    import ipdb; ipdb.set_trace()
 
 
    
