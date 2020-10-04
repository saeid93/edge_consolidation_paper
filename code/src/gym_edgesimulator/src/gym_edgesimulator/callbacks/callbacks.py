import pprint
from unittest.mock import patch

def unsorted_pprint(*args, **kwargs):
    with patch('builtins.sorted', new=lambda l, **_: l):
        orig_pprint(*args, **kwargs)

orig_pprint = pprint.pprint
pprint.pprint = unsorted_pprint

import tensorflow as tf
import numpy as np

# from stable_baselines import SAC
from stable_baselines.common.callbacks import BaseCallback

# model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="/tmp/sac/", verbose=1)

# TODO make it independent of tensorflow
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        summary = tf.Summary(value=[tf.Summary.Value(tag='fraction_of_latency', simple_value=self.training_env.buf_infos[0]['fraction_of_latency'])])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

        summary = tf.Summary(value=[tf.Summary.Value(tag='num_of_consolidated', simple_value=self.training_env.buf_infos[0]['num_of_consolidated'])])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

        return True


class ObjReturnCallback(BaseCallback):
    """
    Custom callback for getting the final objective of the
    algorithm
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(ObjReturnCallback, self).__init__(verbose)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.latency_obj = self.training_env.buf_infos[0]['fraction_of_latency']
        self.consolidation_obj = self.training_env.buf_infos[0]['num_of_consolidated']
        # return self.latency_obj, self.consolidation_obj
        # print("Training Ended!")


class DebugCallback(BaseCallback):
    """
    Custom callback for debugging output.
    """
    def __init__(self, verbose=0, with_fig=False):
        # self.is_tb_set = False
        super(DebugCallback, self).__init__(verbose)
        self.with_fig = with_fig

    def _on_step(self) -> bool:
        # if self.training_env.buf_infos[0]['state']==2:
        print(f"n_calls: {self.n_calls}")
        pprint.pprint(self.training_env.buf_infos[0])
        if self.with_fig:
            self.training_env.buf_infos[0]['simulator'].visualize().savefig(f"{self.n_calls}.png")
        print('--------------------------')
        return True