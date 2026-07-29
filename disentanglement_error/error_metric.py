import json

import numpy as np
from sklearn.model_selection import train_test_split

from disentanglement_error.decreasing_dataset import decreasing_dataset_experiment, decreasing_dataset_experiment_torch
from disentanglement_error.label_noise import label_noise_experiment, label_noise_experiment_torch
from disentanglement_error.util import Config, RunResults, CustomJsonEncoder


def calculate_disentanglement_error(x_train, y_train, disentangling_model, x_test=None, y_test=None, kw_config=None, return_json=True):
    if not kw_config:
        config = Config()
    else:
        config = Config(**kw_config)

    if x_test is None or y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    ua_Ue_corrs = np.empty(config.n_runs)
    ue_Ue_corrs = np.empty(config.n_runs)
    ua_Ua_corrs = np.empty(config.n_runs)
    ue_Ua_corrs = np.empty(config.n_runs)
    results = RunResults()
    for run in range(config.n_runs):
        (corr_ua_Ue, corr_ue_Ue), decreasing_dataset_result  = decreasing_dataset_experiment(x_train, y_train, x_test, y_test, disentangling_model, config)
        (corr_ua_Ua, corr_ue_Ua),  label_noise_result = label_noise_experiment(x_train, y_train, x_test, y_test, disentangling_model, config)
        results.label_noise_results.append(label_noise_result)
        results.decreasing_dataset_results.append(decreasing_dataset_result)
        ua_Ua_corrs[run] = corr_ua_Ua
        ua_Ue_corrs[run] = corr_ua_Ue
        ue_Ua_corrs[run] = corr_ue_Ua
        ue_Ue_corrs[run] = corr_ue_Ue

    disentanglement_error = 1 / (1 + np.sum(config.term_weights)) * (
            np.abs(ua_Ua_corrs - 1).mean() +
            config.term_weights[0] * np.abs(ua_Ua_corrs - 1).mean() +
            config.term_weights[1] * np.abs(ua_Ue_corrs).mean() +
            config.term_weights[2] * np.abs(ue_Ua_corrs).mean()
    )
    correlations = {"ua_Ua_corr": ua_Ua_corrs.mean(),
                                       "ue_Ue_corr": ue_Ue_corrs.mean(),
                                       "ua_Ue_corr": ua_Ue_corrs.mean(),
                                       "ue_Ua_corr": ue_Ua_corrs.mean(),
                                       }

    if return_json:
        results_json = json.dumps(results, cls=CustomJsonEncoder)
        config_json = json.dumps(config, cls=CustomJsonEncoder)
        return disentanglement_error, correlations, results_json, config_json
    else:
        return disentanglement_error, correlations


def calculate_disentanglement_error_torch(train_dataset, val_dataset, disentangling_model, batch_size, num_workers, kw_config=None, return_json=True):
    if not kw_config:
        config = Config()
    else:
        config = Config(**kw_config)

    ua_Ue_corrs = np.empty(config.n_runs)
    ue_Ue_corrs = np.empty(config.n_runs)
    ua_Ua_corrs = np.empty(config.n_runs)
    ue_Ua_corrs = np.empty(config.n_runs)

    results = RunResults()
    for run in range(config.n_runs):
        (corr_ua_Ue, corr_ue_Ue), decreasing_dataset_result  = decreasing_dataset_experiment_torch(train_dataset, val_dataset, disentangling_model, config, batch_size, num_workers)
        (corr_ua_Ua, corr_ue_Ua), label_noise_result = label_noise_experiment_torch(train_dataset, val_dataset, disentangling_model, config, batch_size, num_workers)
        results.label_noise_results.append(label_noise_result)
        results.decreasing_dataset_results.append(decreasing_dataset_result)
        ua_Ua_corrs[run] = corr_ua_Ua
        ua_Ue_corrs[run] = corr_ua_Ue
        ue_Ua_corrs[run] = corr_ue_Ua
        ue_Ue_corrs[run] = corr_ue_Ue

    disentanglement_error = 1 / (1 + np.sum(config.term_weights)) * (
            np.abs(ua_Ua_corrs - 1).mean() +
            config.term_weights[0] * np.abs(ua_Ua_corrs - 1).mean() +
            config.term_weights[1] * np.abs(ua_Ue_corrs).mean() +
            config.term_weights[2] * np.abs(ue_Ua_corrs).mean()
    )
    correlations = {"ua_Ua_corr": ua_Ua_corrs.mean(),
        "ue_Ue_corr": ue_Ue_corrs.mean(),
        "ua_Ue_corr": ua_Ue_corrs.mean(),
        "ue_Ua_corr": ue_Ua_corrs.mean(),
    }

    if return_json:
        results_json = json.dumps(results, cls=CustomJsonEncoder)
        config_json = json.dumps(config, cls=CustomJsonEncoder)
        return disentanglement_error, correlations, results_json, config_json
    else:
        return disentanglement_error, correlations
