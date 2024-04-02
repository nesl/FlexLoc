import numpy as np
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset
import configs.data_configs as data_configs
import configs.train_configs as train_configs
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from cache_datasets import cache_data
from models.GTDM_Model import GTDM, GTDM_Early
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description='Enter the desired seed')
    parser.add_argument("seedVal")
    args = parser.parse_args()
    seedVal = int(args.seedVal)
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(seedVal)
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    np.random.seed(seedVal)

    model = GTDM_Early(train_configs.adapter_hidden_dim, valid_mods=data_configs.valid_mods, valid_nodes = data_configs.valid_nodes)


    # Get current date and time to create new training directory within ./logs/ to store model weights
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y %H_%M_%S")
    os.mkdir('./logs/' + dt_string)
    cache_data() # Runs cacher from the data_configs.py file, will convert hdf5 to pickle if not already done
    
    #PickleDataset inherits from a Pytorch Dataset, creates train and val datasets
    trainset = PickleDataset(data_configs.cache_dir + 'train', data_configs.valid_mods, data_configs.valid_nodes)
    valset = PickleDataset(data_configs.cache_dir + 'val', data_configs.valid_mods, data_configs.valid_nodes)
    batch_size = train_configs.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.pose_expand['zed_camera_left'].parameters():
    #     param.requires_grad = True
    # for param in model.output_head.parameters():
    #     param.requires_grad=True
    # Create the overall model and load on appropriate device
    #model.pose_expand.apply(init_weights)
    #model.load_state_dict(torch.load('./logs/MSELoss_VIT/last.pt'))
    model.to(device)

    #Establish from training parameters

    N_EPOCHS = train_configs.num_epochs
    LR = train_configs.learning_rate

    optimizer = Adam(model.parameters(), lr=LR)
    writer = SummaryWriter(log_dir='./logs/' + dt_string) # Implement tensorboard
    best_val_loss = 0
    mseloss = nn.MSELoss()
    
    # Training loop
    for epoch in trange(N_EPOCHS, desc="Training"):

        batch_num = 0
        epoch_train_loss = 0
        for batch in train_dataloader:
            batch_num += 1
            optimizer.zero_grad()
            train_loss = 0.0

            # Each batch is a dictionary containing all the sensor data, and the ground truth positions
            data, gt_pos = batch['data'], batch['gt_pos']
            # Data itself is a dictionary with keys ('modality', 'node') that points to data of dimension batch_size
            gt_pos = gt_pos.to(device)
            for key in data.keys():
                if ('mocap' in key):
                    continue
                data[key] = data[key].to(device)
            
            # Perform forward pass
            batch_results = model(data) #Dictionary
            # key is still ('modality', 'node') with a distribution estimated by the model
            for key in batch_results.keys():
                for i in range(len(batch_results[key]['dist'])):
                    # TODO Currently 2D, also introduce hybrid training, use MSE to help convergence at start then use NLL
        
                    loss_mse = mseloss(torch.squeeze(batch_results[key]['dist'][i].mean), gt_pos[i][:, [0, 2]])
                    pos_neg_log_probs =  -batch_results[key]['dist'][i].log_prob(gt_pos[i][:, [0, 2]]) # Computes NLL loss for each node/modality combo
                    train_loss += pos_neg_log_probs + 0.05 * loss_mse # Accumulate all the losses into the batch loss
            train_loss /= (batch_size * len(batch_results.keys())) # Normalize wrt batch size and number of modality node combinations

            with torch.no_grad():
                # Print one sample from the batch to see prediction result and loss
                print('Batch Number', batch_num)
                key = 'early_fusion'
                print('Estimate', batch_results[key]['dist'][0].mean.data, " with cov ",  batch_results[key]['dist'][0].covariance_matrix.data)
                sample_mse_loss =  mseloss(torch.squeeze(batch_results[key]['dist'][0].mean), gt_pos[0][:, [0, 2]])
                sample_nll_loss =  -batch_results[key]['dist'][0].log_prob(gt_pos[0][:, [0, 2]])
                print('\tGT', gt_pos[0], 'with loss', sample_nll_loss + 0.05 * sample_mse_loss)
                #print('\tGT', gt_pos[0], 'with loss', mseloss(batch_results['img', 'node_1']['dist'][0].mean, gt_pos[0][:, 0:2]))
                print('-------------------------------------------------------------------------------------------------------------------------------')
                epoch_train_loss += train_loss # Accumulate batch loss into overall epoch loss
            # Backprop and update
            train_loss.backward()
            optimizer.step() 
        
        print('TRAIN LOSS', epoch_train_loss / batch_num)
        writer.add_scalar("Training loss", epoch_train_loss / batch_num, epoch)

        # Compute validation loss
         # Compute validation loss
        with torch.no_grad():
            model.eval()
            val_loss_arr = []
            for i, batch in enumerate(val_dataloader):
                # Once again, get data and ground truth from batch 
                data, gt_pos = batch['data'], batch['gt_pos']
                gt_pos = gt_pos.to(device)
                for key in data.keys():
                    if ('mocap' in key):
                        continue
                    data[key] = data[key].to(device)
                
                batch_results = model(data) # Run the model

                for key in batch_results.keys():
                    for i in range(len(batch_results[key]['dist'])):
                        sample_mse_loss = mseloss(batch_results[key]['dist'][i].mean, gt_pos[i][:, [0,2]])
                        pos_neg_log_probs = -batch_results[key]['dist'][i].log_prob(gt_pos[i][:, [0,2]]) # Computes loss for each node/modality combo
                        val_loss_arr.append(sample_mse_loss)
            
            median_val_loss = sorted(val_loss_arr)[int(len(val_loss_arr)/2)]
            writer.add_scalar("Validation loss", median_val_loss, epoch)
            model.train()
            # Save model with best validation loss and/or most recent
            if train_configs.save_best_model:
                if (epoch == 0):
                    best_val_loss = median_val_loss
                if (median_val_loss < best_val_loss):
                    torch.save(model.state_dict(), './logs/' + dt_string + '/best_val.pt')
                    best_val_loss = median_val_loss
            if epoch % train_configs.save_every_X_model == 0:
                torch.save(model.state_dict(), './logs/' + dt_string + '/last.pt')
                

if __name__ == '__main__':
    main()

