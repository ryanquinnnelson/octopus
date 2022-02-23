"""
All things related to wandb.
"""
__author__ = 'ryanquinnnelson'

import logging
import subprocess

import pandas as pd

import octopus.utilities.fileutilities as fu


class WandbConnector:
    """
    Defines an object which manages the connection to wandb.
    """

    def __init__(self, wandb_dir, entity, run_name, project, notes, tags, mode, config):
        """
        Initialize wandb connector.
        Args:
            wandb_dir (str): fully-qualified directory where kaggle internal files will be stored (provides user control
        over their directories)
            entity (str): name of the wandb user account
            run_name (str): unique name identifying this run
            project (str): project name where runs will be stored on wandb
            notes (str): notes to store with this run
            tags (List): list of strings representing tags for wandb
            config (Dict): dictionary of all hyperparameter configurations set for this run
        """
        logging.info('Initializing wandb connector...')
        self.wandb_dir = wandb_dir
        self.entity = entity
        self.run_name = run_name
        self.project = project
        self.notes = notes
        self.tags = tags
        self.mode = mode  # set mode to offline if online mode is failing
        self.config = config
        self.wandb_config = None
        self.run = None

    def initialize_wandb(self):
        """
        Initialize wandb connection. Define reinitialization to True.
        Returns:wandb.config object
        """

        logging.info('Initializing wandb...')

        # ensure wandb_dir exists
        fu.create_directory(self.wandb_dir)

        import wandb
        self.run = wandb.init(dir=self.wandb_dir,
                              name=self.run_name,
                              project=self.project,
                              notes=self.notes,
                              tags=self.tags,
                              config=self.config,
                              reinit=True,
                              mode=self.mode)

        # save configuration
        self.wandb_config = wandb.config

    def watch(self, model):
        """
         Initialize wandb logging of the model to log weight histograms.
        Args:
            model (nn.Module): model to watch
        Returns:None
        """

        logging.info('Watching model with wandb...')
        import wandb
        wandb.watch(model)

    def log_stats(self, stats_dict):
        """
         Upload given stats to wandb.
        Args:
            stats_dict (Dict): dictionary of stats to upload to wandb
        Returns:None
        """

        import wandb
        wandb.log(stats_dict)

    def update_best_model(self, updates_dict):
        """
        Update the best model stored in wandb.
        Args:
            updates_dict (Dict): dictionary of stats to upload to wandb
        Returns:None
        """

        import wandb

        for key, value in updates_dict.items():
            wandb.run.summary[key] = value

    def concatenate_run_metrics(self, metric, epoch):
        """
        Pull values for all runs for a given (epoch,metric) and combine runs into a single DataFrame.
        Args:
            metric (str): metric to extract
            epoch (int): epoch to extract
        Returns:DataFrame with (name, epoch, metric) columns and a row for each run.
        """

        runs = self.pull_runs()
        valid_run_metrics = []
        for run in runs:
            name = run.name
            metrics_df = run.history()
            cols = list(metrics_df.columns)

            if not metrics_df.empty and 'epoch' in cols and metric in cols:
                # add run name to columns
                metrics_df['name'] = name
                valid_run_metrics.append(metrics_df[['name', 'epoch', metric]])

        # pool metrics for all runs
        if len(valid_run_metrics) > 0:
            df = pd.concat(valid_run_metrics)
            df = df[df['epoch'] == epoch]
        else:
            df = pd.DataFrame(columns=['name', 'epoch', metric])  # empty dataframe

        return df

    def get_best_value(self, metric, epoch, best_is_max):
        """
        Gather all run metrics for a given (epoch, metric) and determine the best run for the data returned.
        Args:
            metric (str): metric to extract
            epoch (int): epoch number to extract
            best_is_max (Boolean): True if the best value will be the maximum value
        Returns:Tuple (str,float) representing the name of the best run and value of the metric of the best run
        """

        # gather the metrics for each valid run
        metrics_df = self.concatenate_run_metrics(metric, epoch)

        # temp output
        a = metrics_df[metrics_df['epoch'] == epoch].sort_values(metric)
        logging.info(f'Gathered metrics for epoch {epoch} from wandb:\n{a}')

        if metrics_df.empty:
            best_name, best_val = None, None
        else:
            best_name, best_val = self.calculate_best_value(metrics_df, metric, epoch, best_is_max)

        return best_name, best_val

    def pull_runs(self):
        """
        Pull all run data for the entity and project defined in this connector.
        Returns: wandb runs object
        """

        import wandb
        api = wandb.Api()

        runs = api.runs(f'{self.entity}/{self.project}')
        return runs

    def calculate_best_value(self, metrics_df, metric, epoch, best_is_max):

        """
        Given a DataFrame of runs, determine the best run.
        Args:
            metrics_df (DataFrame): DataFrame containing columns (name,epoch,metric) and a row for each run.
            metric (str): metric to extract
            epoch (int): epoch number to extract
            best_is_max (Boolean): True if the best value will be the maximum value
        Returns:Tuple (str,float) representing the name of the best run and value of the metric of the best run
        """

        if best_is_max:
            # find the max value of the metric for each epoch
            bests = metrics_df.groupby(by='epoch').max().reset_index()
        else:
            # find the min
            bests = metrics_df.groupby(by='epoch').min().reset_index()

        # choose the best value for the epoch we care about
        best_val = bests[bests['epoch'] == epoch][metric].item()
        best_name = metrics_df[metrics_df[metric] == best_val]['name'].iloc[0]
        return best_name, best_val

    def login(self):

        """
        Log into wandb.
        Returns: None
        """

        logging.info('Logging into wandb...')

        import wandb
        wandb.login()
