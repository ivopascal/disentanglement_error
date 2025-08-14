import json

import numpy as np
from sklearn.model_selection import train_test_split

from disentanglement_error.decreasing_dataset import decreasing_dataset_experiment
from disentanglement_error.label_noise import label_noise_experiment
from disentanglement_error.util import Config, RunResults, CustomJsonEncoder


def calculate_disentanglement_error(x_train, y_train, disentangling_model, x_test=None, y_test=None, kw_config=None, return_json=True):
    config = Config(**kw_config)

    if x_test is None or y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    disentanglement_errors = []
    results = RunResults()
    for run in range(config.n_runs):
        decreasing_dataset_error, decreasing_dataset_result  = decreasing_dataset_experiment(x_train, y_train, x_test, y_test, disentangling_model, config)
        label_noise_error, label_noise_result = label_noise_experiment(x_train, y_train, x_test, y_test, disentangling_model, config)
        results.label_noise_results.append(label_noise_result)
        results.decreasing_dataset_results.append(decreasing_dataset_result)
        disentanglement_errors.append((decreasing_dataset_error + label_noise_error) / 4)

    if return_json:
        results_json = json.dumps(results, cls=CustomJsonEncoder)
        config_json = json.dumps(config, cls=CustomJsonEncoder)
        return np.mean(disentanglement_errors), results_json, config_json
    else:
        return np.mean(disentanglement_errors)