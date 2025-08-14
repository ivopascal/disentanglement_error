import numpy as np
from scipy.stats import pearsonr
from sklearn.utils import shuffle

from disentanglement_error.util import ExperimentResults


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
