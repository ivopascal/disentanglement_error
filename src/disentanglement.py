from typing import List

from sklearn.model_selection import train_test_split
import numpy as np

from src.decreasing_dataset import decreasing_dataset_error
from src.label_noise import label_noise_error
from dataclasses import dataclass, field


@dataclass
class Config:
    dataset_sizes: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
    label_noises: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_runs: int = 5


def disentanglement_error(X_train, y_train, disentangling_model, X_test=None, y_test=None, kw_config=None):
    config = Config(**kw_config)

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    disentanglement_errors = []
    for run in range(config.n_runs):
        epistemic_experiment_error = decreasing_dataset_error(X_train, y_train, X_test, y_test, disentangling_model, config)
        aleatoric_experiment_error = label_noise_error(X_train, y_train, X_test, y_test, disentangling_model, config)
        disentanglement_errors.append((epistemic_experiment_error + aleatoric_experiment_error) / 4)

    return np.mean(disentanglement_errors)
