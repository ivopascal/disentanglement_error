from typing import Tuple

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from disentanglement_error.util import ExperimentResults, AverageMeter, Summary


def decreasing_dataset_experiment(x_train, y_train, x_test, y_test, model, config):
    dataset_sizes = config.dataset_sizes

    x_train, y_train = shuffle(x_train, y_train)

    experiment_results = ExperimentResults()
    for dataset_size in dataset_sizes:
        x_train_small, y_train_small = create_subsampled_dataset(x_train, y_train, dataset_size)
        
        model.fit(x_train_small, y_train_small)

        predictions, aleatorics, epistemics = model.predict_disentangling(x_test)

        score = model.score(y_test, predictions)

        experiment_results.scores.append(score)
        experiment_results.aleatorics.append(aleatorics.mean())
        experiment_results.epistemics.append(epistemics.mean())


    aleatoric_pcc, _ = pearsonr(experiment_results.aleatorics, experiment_results.scores)
    epistemic_pcc, _ = pearsonr(experiment_results.epistemics, experiment_results.scores)

    return np.abs(aleatoric_pcc - 0) + np.abs(epistemic_pcc - 1), experiment_results

def create_subsampled_dataset(x_train, y_train, dataset_size):
    X_train_subs = []
    y_train_subs = []

    # We might just use Stratified Cross-validation for this...
    for y_value in np.unique(y_train):
        n_samples_per_class = int(np.sum((y_train == y_value)) * dataset_size)
        if n_samples_per_class == 0:
            n_samples_per_class = 1
        X_train_subs.append(x_train[y_train == y_value][:n_samples_per_class])
        y_train_subs.append(y_train[y_train == y_value][:n_samples_per_class])

    X_train_sub = np.concatenate(X_train_subs)
    y_train_sub = np.concatenate(y_train_subs)
    X_train_sub, y_train_sub = shuffle(X_train_sub, y_train_sub)

    return X_train_sub, y_train_sub



def create_subset_dataloaders(train_dataset, val_dataset, percentage, batch_size, workers) -> Tuple[DataLoader, DataLoader]:
    indices = torch.randperm(len(train_dataset.samples))[:int(len(train_dataset.samples) * percentage)]
    train_sampler = torch.utils.data.SubsetRandomSampler(indices)
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


def decreasing_dataset_experiment_torch(train_dataset, val_dataset, model, config, batch_size, workers, train_only=False):
    dataset_sizes = config.dataset_sizes

    experiment_results = ExperimentResults()
    for dataset_size in dataset_sizes:
        use_accel = True
        train_dataloader, val_dataloader = create_subset_dataloaders(train_dataset, val_dataset, dataset_size, batch_size=batch_size, workers=workers)

        model.fit(train_dataloader, val_dataloader, dataset_size=dataset_size)

        if not train_only:
            accuracy_meter = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
            aleatoric_meter = AverageMeter('Ale', use_accel, ':6.2f', Summary.AVERAGE)
            epistemic_meter = AverageMeter('Epi', use_accel, ':6.2f', Summary.AVERAGE)
            for i, (images, target) in enumerate(val_dataloader):
                predictions, aleatorics, epistemics = model.predict_disentangling(images)

                score = model.score(target, predictions.cpu())

                accuracy_meter.update(score[0], images.size(0))
                aleatoric_meter.update(aleatorics.mean().cpu(), images.size(0))
                epistemic_meter.update(epistemics.mean().cpu(), images.size(0))

            experiment_results.scores.append(accuracy_meter.avg)
            experiment_results.aleatorics.append(aleatoric_meter.avg)
            experiment_results.epistemics.append(epistemic_meter.avg)
        else:
            experiment_results.scores.append(0.5)
            experiment_results.aleatorics.append(0.5)
            experiment_results.epistemics.append(0.5)

    aleatoric_pcc, _ = pearsonr(experiment_results.aleatorics, experiment_results.scores)
    epistemic_pcc, _ = pearsonr(experiment_results.epistemics, experiment_results.scores)

    return np.abs(aleatoric_pcc - 0) + np.abs(epistemic_pcc - 1), experiment_results