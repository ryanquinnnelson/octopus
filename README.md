# octopus
Author: Ryan Nelson

```text
               _---_
             /       \
            |         |
    _--_    |         |    _--_
   /__  \   \  0   0  /   /  __\
      \  \   \       /   /  /
       \  -__-       -__-  /
   |\   \    __     __    /   /|
   | \___----         ----___/ |
   \                           /
    --___--/    / \    \--___--
          /    /   \    \
    --___-    /     \    -___--
    \_    __-         -__    _/
      ----               ----
       
        O  C  T  O  P  U  S
```

## Introduction

`octopus` is a python module that standardizes the execution of deep learning pipelines using `pytorch` and  `wandb`. Module behavior is controlled using a configuration file (ConfigParser format).



## Using `octopus`
```
# add the path to the octopus python package
sys.path.append('/path/to/octopus')
from octopus.octopus import Octopus

# your code to parse the configuration and build the required handlers

# run octopus
octopus = Octopus(config_file, config, datasethandler, phasehandler, modelhandler, optimizerhandler, schedulerhandler)
octopus.run()
```

## Handler requirements
The required handlers must have the following interface. 
```
# dataset handler
datasethandler.get_train_dataset(config) -> torch.utils.data.Dataset
datasethandler.get_val_dataset(config) -> torch.utils.data.Dataset
datasethandler.get_test_dataset(config) -> torch.utils.data.Dataset


# phase handler
phasehandler.get_train_phase(devicehandler, dataloader, config) -> Training
phasehandler.get_val_phase(devicehandler, dataloader, config) -> Validation
phasehandler.get_test_phase(devicehandler, dataloader, config, output_dir) -> Testing

Training.run_epoch(epoch, num_epochs, models, optimizers) -> Dict  # {'train_loss': 35.54 }
Validation.run_epoch(epoch, num_epochs, models) -> Dict
Testing.run_epoch(epoch, num_epochs, models) -> Dict


# model handler
modelhandler.get_models(config) -> 
(Collection[torch.nn.Module], Collection[String])  # (models, model_names)


# optimizer handler
optimizerhandler.get_optimizers(models, config) -> 
(Collection[torch.optim], Collection[String])  # (optimizers, optimizer_names)


# scheduler handler
schedulerhandler.get_schedulers(optimizers, config) -> 
(Collection[torch.optim as optim], Collection[String])  # (schedulers, scheduler_names)


```

## configuration file requirements
The following entries are required in the config file for the `octopus` framework to function. All additional configurations should be customized based on your requirements.
```
[DEFAULT]
run_name = name-of-run-in-wandb


[debug]
debug_path = /home/ubuntu


[wandb]
wandb_dir = /home/ubuntu/wandb
entity = ryanquinnnelson
project = testproject
notes = Image Segmentation using GANs
tags = octopus,GAN,DAN
mode = online
config_sections_to_track=dataloader,model,hyperparameters


[output]
output_dir = /home/ubuntu/output


[checkpoint]
checkpoint_dir = /data/checkpoints
checkpoint_cadence = 5
delete_existing_checkpoints = True
load_from_checkpoint=False
checkpoint_file =  None


[dataloader]
num_workers=8
pin_memory=True
batch_size=10


[hyperparameters]
num_epochs = 4

```


## Additional functionality
This package contains two scripts to perform common tasks on AWS EC2 instances.

Mounts a NFS drive that comes with an EC2 instance.
```commandline
bash /path/to/octopus/bin/mount_drive
```


Installs and logs into wandb, both of which are required to use wandb in octopus. Also required so sweeps can be performed.
```commandline
bash /path/to/octopus/bin/setup_wandb
```