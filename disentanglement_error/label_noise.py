import numpy as np
from scipy.stats import pearsonr
from sklearn.utils import shuffle

from disentanglement_error.util import ExperimentResults


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



