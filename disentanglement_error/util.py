import dataclasses
import json
from enum import Enum

import torch.distributed as dist

import pandas as pd
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


def json_results_to_df(json_results, json_config):
    result = json.loads(json_results)
    config = json.loads(json_config)

    dfs = []
    for run_index, experiment_result in enumerate(result['label_noise_results']):
        df = pd.DataFrame(experiment_result).assign(Run_Index=run_index, Experiment="Label Noise",
                                                    Percentage=config['label_noises'])
        dfs.append(df)
    for run_index, experiment_result in enumerate(result['decreasing_dataset_results']):
        df = pd.DataFrame(experiment_result).assign(Run_Index=run_index, Experiment="Decreasing Dataset",
                                                    Percentage=config['dataset_sizes'])
        dfs.append(df)
    df = pd.concat(dfs)
    return df


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return json.JSONEncoder.default(self, obj)


@dataclass
class Config:
    dataset_sizes: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
    label_noises: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_runs: int = 5

@dataclass
class ExperimentResults:
    scores: List[float] = field(default_factory=lambda: [])
    aleatorics: List[float] = field(default_factory=lambda: [])
    epistemics: List[float] = field(default_factory=lambda: [])

@dataclass
class RunResults:
    label_noise_results: List[ExperimentResults] = field(default_factory=lambda: [])
    decreasing_dataset_results: List[ExperimentResults] = field(default_factory=lambda: [])



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)
