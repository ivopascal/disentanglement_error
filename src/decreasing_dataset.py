import numpy as np
from scipy.stats import pearsonr
from sklearn.utils import shuffle

def decreasing_dataset_error(x_train, y_train, x_test, y_test, model, config):
    dataset_sizes = config.dataset_sizes

    x_train, y_train = shuffle(x_train, y_train)

    aleatoric_means = []
    epistemic_means = []
    scores = []
    for dataset_size in dataset_sizes:
        x_train_small, y_train_small = create_subsampled_dataset(x_train, y_train, dataset_size)
        
        model.fit(x_train_small, y_train_small)

        predictions, aleatorics, epistemics = model.predict_disentangling(x_test)

        score = model.score(y_test, predictions)
        aleatoric_means.append(np.mean(aleatorics))
        epistemic_means.append(np.mean(epistemics))
        scores.append(score)


    aleatoric_pcc, _ = pearsonr(aleatoric_means, scores)
    epistemic_pcc, _ = pearsonr(epistemic_means, scores)

    return np.abs(aleatoric_pcc - 0) + np.abs(epistemic_pcc - 1)

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
