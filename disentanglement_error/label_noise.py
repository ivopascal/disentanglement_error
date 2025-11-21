import random

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.utils import shuffle

from disentanglement_error.util import ExperimentResults, AverageMeter, Summary


def label_noise_experiment(x_train, y_train, x_test, y_test, model, config):
    noises = config.label_noises

    experiment_results = ExperimentResults()
    for noise in noises:
        X_train_noisy, y_train_noisy = partial_shuffle_dataset(x_train, y_train, percentage=noise)
        X_test_noisy, y_test_noisy = partial_shuffle_dataset(x_test, y_test, percentage=noise)

        model.fit(X_train_noisy, y_train_noisy)

        predictions, aleatorics, epistemics = model.predict_disentangling(X_test_noisy)

        score = model.score(y_test_noisy, predictions)
        experiment_results.scores.append(score)
        experiment_results.aleatorics.append(np.mean(aleatorics))
        experiment_results.epistemics.append(np.mean(epistemics))

    aleatoric_pcc, _ = pearsonr(experiment_results.aleatorics, experiment_results.scores)
    epistemic_pcc, _ = pearsonr(experiment_results.epistemics, experiment_results.scores)

    return np.abs(aleatoric_pcc - 1) + np.abs(epistemic_pcc - 0), experiment_results


def partial_shuffle_dataset(x, y, percentage):
    x_noisy, y_noisy = shuffle(x, y)
    np.random.shuffle(y_noisy[:int(len(y_noisy) * percentage)])
    x_noisy, y_noisy = shuffle(x_noisy, y_noisy)
    return x_noisy, y_noisy




def label_noise_experiment_torch(train_dataset, val_dataset, model, config, batch_size, workers, train_only=False):
    noises = config.label_noises

    experiment_results = ExperimentResults()
    for noise in noises:
        use_accel = True
        train_dataloader = create_partial_shuffle_dataloader_torch(train_dataset, percentage=noise, batch_size=batch_size, workers=workers)
        val_dataloader = create_partial_shuffle_dataloader_torch(val_dataset, percentage=noise, batch_size=batch_size, workers=workers)

        model.fit(train_dataloader, val_dataloader, label_noise=noise)

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

    return np.abs(aleatoric_pcc - 1) + np.abs(epistemic_pcc - 0), experiment_results


def create_partial_shuffle_dataloader_torch(dataset, percentage, batch_size, workers):
    indices = np.random.choice(np.arange(len(dataset.samples)), size=int(percentage * len(dataset.samples)), replace=False)

    subset_to_shuffle = [dataset.samples[index] for index in indices]
    random.shuffle(subset_to_shuffle)

    for i, index in enumerate(indices):
        dataset.samples[index] = (dataset.samples[index][0], subset_to_shuffle[i][1])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None)

    return data_loader

