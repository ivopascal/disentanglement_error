import dataclasses
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import List

import numpy as np

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
