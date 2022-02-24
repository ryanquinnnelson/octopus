# octopus
Framework for training deep learning models


## Run single run
Executes `octopus` for a single configuration file.

One time setup
```commandline
bash /path/to/octopus/bin/mount_drive
bash /path/to/octopus/bin/setup_environment
```
For each run
```commandline
python /path/to/octopus/run_octopus.py --filename=/path/to/octopus/configs/remote/config001.txt
```

## Run sweep
wandb conducts a search over your hyperparameters. Set the configuration file within the sweep yaml file.

One time setup
```commandline
bash /path/to/octopus/bin/mount_drive
bash /path/to/octopus/bin/setup_environment
```
For each sweep
```commandline
wandb sweep /path/to/octopus/sweeps/remote/sweep001.yaml

# execute wandb agent as specified by wandb
```