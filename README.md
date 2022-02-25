# octopus
Framework for training deep learning models


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
