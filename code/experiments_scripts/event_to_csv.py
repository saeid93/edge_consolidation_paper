# code from: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py

import tensorflow as tf
import glob
import os
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import click
import pprint


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    
    Parameters
    ----------
    path : str
        path to tensorflow log file
    
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def to_csv(logdir_or_logfile: str, out_dir: str):
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        # pp.pprint("Found tensorflow logs to process:")
        # pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths)
        # pp.pprint("Head of created dataframe")
        # pp.pprint(all_logs.head())

        os.makedirs(out_dir, exist_ok=True)

        print("saving to csv file")
        
        # extract desired part of the results
        fraction_of_latency = all_logs[all_logs['metric']=="fraction_of_latency"][['step','value']]
        num_of_consolidated = all_logs[all_logs['metric']=="num_of_consolidated"][['step','value']]
        episode_reward = all_logs[all_logs['metric']=="episode_reward"][['step','value']]

        # path to save results in selected files
        out_file_l = os.path.join(out_dir, "fraction_of_latency.csv")
        out_file_c = os.path.join(out_dir, "num_of_consolidated.csv")
        out_file_r = os.path.join(out_dir, "episode_reward.csv")

        # save results in selected files
        fraction_of_latency.to_csv(out_file_l, index=None)
        num_of_consolidated.to_csv(out_file_c, index=None)
        episode_reward.to_csv(out_file_r, index=None)
    else:
        print("No event paths have been found.")
