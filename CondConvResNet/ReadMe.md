# IoBT Tracking

## Installing dependencies
Utilize the provided requirements.txt file to install the required dependencies (there may be a lot more than necessary)

## Structure of the Project

### Configurations
Data and train configurations are stored in the configs/ folder
```data_configs.py``` holds the location of the hdf5 files and the location to cache
```train_configs.py``` consists of some parameters to change for training

### Models

You can find all the different neural network architectures within models/
```GTDM_Model.py``` assembles all the different components to create one unified model used during training and testing. Note that currently we are using resnets with pre-trained weights that are frozen to verify model accuracy. 

### Training and Testing 

The training process can be executed by running ```python3 train.py```. Within the train.py file, we perform caching, create the appropriate datasets and dataloaders, create the GTDM model, and begin the training process according to the specified parameters. We create a folder under ```logs/``` holding all the model weights, and periodically log to Tensorboard under ```runs/``` to monitor training progress. Currently, we are only training on camera

The testing process can be executed with ```python3 test.py```. In this file, we provide the file path to the appropriate folder under ```logs/``` to load weights and use for testing. A video file will be generated in the same folder.