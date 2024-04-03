# FlexLoc

## Create Environment
1. Clone the repository to any directory
2. Create a conda environment with the following command `conda create --env flexloc python=3.10`
3. In the `CondConvResNet` directory, there is a `requirements.txt` with the necessary packages
```
cd CondConvResNet
pip3 install -r requirements.txt
```
4. Separately install pytorch
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```




## Run Small Scale Test

We include 30 files from our test set to run a small scale test verifying that all the libraries are correctly installed.

In either conditional convolution (CondConvResNet) or conditional layer normalization (Final_CLN_Resnet) folders, run 
```
cd CondConvResNet OR cd Final_CLN_Resnet
python3 batch_test.py --folder 1 --checkpoint best_val.pt
```
to run a small scale test utilizing our provided checkpoints.

In the `logs` folder, under the folder `1`, it generates two .txt files. `predictions.txt` contains the predicted coordinates vs. the ground truth coordinates, while `test_loss.txt` contains the evaluation metrics. We utilize the Average Distance metric for our evaluations.


## Run Large Scale Test

1. Download data from this [Google Drive Link](https://drive.google.com/file/d/1t8fxeyyrl_0TaG-YCABYgbP7Nt_KXHSY/view?usp=sharing).
2. After unzipping the data, there will be 5 different test folders (`test29`, `test70`, `test73`, `test74`, `test77`) each containing 500 cached pickle files representing ~30 seconds of data
3. Place these folders into the top level respository directory
4. Rename a given viewpoint's folder name to `test`. For example, if we want to evaluate the model on viewpoint 29, rename `test29` to `test`.
5. Navigate to the appropriate CLN or CondConv directory, and run `python3 batch_test.py --folder 1 --checkpoint best_val.pt`. The results will be under `test_loss.txt` in `logs/1`
6. Revert back to the original name of the test data, e.g., rename `test` back to `test29`.
7. You can test on other viewpoints by repeating steps 4-6. Note that successive calls to the `batch.test.py` script will _append_ to the `test_loss.txt` file instead of overwriting previous data.

## Train the Model
1. Refer to the [GTDM work](https://github.com/nesl/GDTM) to download the entire dataset and run necessary preprocessing
2. Once we have the files in hdf5 form separated into appropriate `train`, `test`, and `val` directories, we must cache these into .pickle files for training. Provide the root directory to these hdf5 files by modifying the `base_root` variable in line 2 of the `configs/data_configs.py`, and adjusting the subsequent `data_root` variables. Adjust the `cache_dir` as you see fit to decide where these files will be cached
3. Running `python3 train.py 100` to begin training with a seed of 100. This will check the specified cache directory for `train`, `test`, and `val` folders containing the pickle files. If they are not found, it will search the specified `data_root` directories for appropriate hdf5 files to begin the process of caching to .pickle. Training will commence after the one-time caching process.
4. The training logs and checkpoints will be saved in `logs` under a folder with the current timestamp
