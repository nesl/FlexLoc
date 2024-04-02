# FlexLoc

## Create Environment
conda create --env flexloc python=3.10
pip3 install -r requirements.txt
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia



Test contains 30 pickle files from view 29, additionally we provide more data at location X


## Run Small Scale Test

In either conditional convolution (CondConv_Resnet) or conditional layer normalization (Final_CLN_Resnet) folders, run 

python3 batch_test.py --folder 1 --checkpoint best_val.pt

In the logs folder, under the folder 1, see the predictions (predictions vs ground truth) and test loss computations. We utilize the Average Distance metric for our evaluations


## Run large scale test

Download data from google drive link


Unzip, will find 5 different test folders each containing the cached pickle files of ~30 seconds of data
