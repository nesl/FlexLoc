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
python3 batch_test.py --folder 1 --checkpoint best_val.pt
```
to run a small scale test utilizing our provided checkpoints.

In the logs folder, under the folder 1, we have two .txt files. `predictions.txt` contains the predicted coordinates vs. the ground truth coordinates, while `test_loss.txt` computations. We utilize the Average Distance metric for our evaluations.


## Run large scale test

Download data from google drive link


Unzip, will find 5 different test folders each containing the cached pickle files of ~30 seconds of data
