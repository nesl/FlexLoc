import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.GTDM_Model import GTDM, GTDM_Early
from PickleDataset import PickleDataset
from tracker import TorchMultiObsKalmanFilter
from video_generator import VideoGenerator
import configs.data_configs as data_configs
import configs.train_configs as train_configs
from cache_datasets import cache_data
import argparse

def computeDist(tensor1, tensor2):
    tensor1 = torch.squeeze(tensor1)
    tensor2 = torch.squeeze(tensor2)
    distance = 0.0
    for i in range(len(tensor1)):
        distance += (tensor1[i] - tensor2[i]) ** 2
    return distance ** 0.5


def main():
    parser = argparse.ArgumentParser(description='Enter the folder')
    parser.add_argument("--folder")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    folder = str(args.folder)
    cache_data()
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = './logs/' + folder + '/'
    # Create model architecture
    model = GTDM_Early(train_configs.adapter_hidden_dim, valid_mods=data_configs.valid_mods, valid_nodes = data_configs.valid_nodes) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    #model.load_state_dict(torch.load(dir_path + str(args.checkpoint)))
    weights_dict = torch.load(dir_path + str(args.checkpoint))
    for name, param in model.state_dict().items():
        if name not in weights_dict or 'final_pos_embedding' in name:
            print(name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model.state_dict()[name].copy_(weights_dict[name])
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = PickleDataset(data_configs.cache_dir + 'test', data_configs.valid_mods, data_configs.valid_nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = 64
    test_dataloader = DataLoader(testset, batch_size = batch_size, shuffle=False)
    # Initialize the kalman filter
    kf = TorchMultiObsKalmanFilter(dt=1, std_acc=1)
    outputs = {}
    outputs['det_means'] = []
    outputs['det_covs'] = []
    outputs['track_means'] = []
    outputs['track_covs'] = []
    total_nll_loss = 0.0
    total_mse_loss = 0.0
    average_dist = 0.0
    mseloss = nn.MSELoss()
    mse_arr = []
    gt_pos_arr = []
    avg_distance_KF = 0.0

    for batch in tqdm(test_dataloader, desc = 'Computing test loss', leave=False):
        with torch.no_grad():
            data, gt_pos = batch['data'], batch['gt_pos']
            gt_pos = torch.squeeze(gt_pos.to(device))
            for key in data.keys():
                if ('mocap' in key):
                    continue
                data[key] = data[key].to(device)

            results = model(data) # Evaluate on test data
   
            all_pred_means = []
            all_pred_covs = []
            # Each modality and node combo has a predicted mean and covariance
            # Even if 3D, we only plot 2D so we take only x and y
            for result in results.values(): # This only runs once even with > 1 batch size
                for i in range(len(torch.squeeze(result['pred_mean']))):
                    if (len(result['pred_mean']) == 1):
                        break

                    predicted_means = torch.squeeze(result['pred_mean'])[i][0:2] # Extract only x and y
                    predicted_covs = torch.squeeze(result['pred_cov'])[i][0:2, 0:2]
                    predicted_means = predicted_means.cpu().detach()
                    predicted_covs = predicted_covs.cpu().detach()
                    gt_pos_arr.append(gt_pos[i, [0, 2]])
                    # Append to output
                    outputs['det_means'].append(predicted_means)
                    outputs['det_covs'].append(predicted_covs)
                    # Calculate loss
                    sample_mse_loss = mseloss(result['dist'][i].mean, gt_pos[i, [0,2]])
                    total_mse_loss += sample_mse_loss

                    average_dist += computeDist(predicted_means, gt_pos[i, [0, 2]])

                    mse_arr.append(sample_mse_loss)
                    total_nll_loss += -result['dist'][i].log_prob(gt_pos[i, [0,2]]) # May be 3D
                    # Perform Kalman Filters
                    kf.update(torch.unsqueeze(predicted_means, 1), [predicted_covs])
                    kf_mean = kf.predict() 
                    kf_cov = kf.P[0:2, 0:2]
                    outputs['track_means'].append(torch.squeeze(kf_mean))
                    outputs['track_covs'].append(kf_cov)
                    avg_distance_KF += computeDist(torch.squeeze(kf_mean).detach().cpu(), gt_pos[i, [0,2]].detach().cpu())
                   


    print("Finished running model inference, generating video with total test loss", total_nll_loss / (len(test_dataloader)))
    f = open(dir_path + "test_loss.txt", "a")
    f.write("\n\n\nComputed NLL Test Loss " + str(total_nll_loss / len(mse_arr)))
    f.write("\nComputed MSE Test Loss " + str(total_mse_loss / len(mse_arr)))
    f.write("\nMedian MSE Loss " + str(sorted(mse_arr)[int(len(mse_arr)/2)]))
    f.write("\nAverage Distance " + str(average_dist / len(mse_arr)))
    f.write("\nAverage KF distance " + str(avg_distance_KF / len(mse_arr)))
    f.close()
    # Write the predictions and the gts
    f = open(dir_path + "predictions.txt", "a")
    f.write('Dets\t\t GT\n')
    for i in range(len(outputs['det_means'])):
        outputs['det_means'][i] = torch.unsqueeze(outputs['det_means'][i], 0)
        outputs['det_covs'][i] = torch.unsqueeze(outputs['det_covs'][i], 0)
        f.write(str(outputs['det_means'][i].cpu().numpy()))
        f.write("\t ")
        f.write(str(gt_pos_arr[i].cpu().numpy()))
        f.write('\n')

      
    #import pdb; pdb.set_trace()
    # Generate the video in the same logs folder as train
    v = VideoGenerator(dir_path, data_configs.valid_mods, data_configs.valid_nodes, testset) # Pass valid mods, nodes, and also hidden layer size
    v.write_video(outputs)


if __name__ == '__main__':
    main()

