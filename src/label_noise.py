import numpy as np
from scipy.stats import pearsonr
from sklearn.utils import shuffle


def label_noise_error(X_train, y_train, X_test, y_test, model, config):
    noises = config.label_noises
    aleatoric_means = []
    epistemic_means = []
    scores = []

    for noise in noises:
        X_train_noisy, y_train_noisy = partial_shuffle_dataset(X_train, y_train, percentage=noise)
        X_test_noisy, y_test_noisy = partial_shuffle_dataset(X_test, y_test, percentage=noise)

        model.fit(X_train_noisy, y_train_noisy)

        predictions, aleatorics, epistemics = model.predict_disentangling(X_test_noisy)

        score = model.score(y_test, predictions)
        aleatoric_means.append(np.mean(aleatorics))
        epistemic_means.append(np.mean(epistemics))
        scores.append(score)

    aleatoric_pcc, _ = pearsonr(aleatoric_means, scores)
    epistemic_pcc, _ = pearsonr(epistemic_means, scores)

    return np.abs(aleatoric_pcc - 1) + np.abs(epistemic_pcc - 0)


def partial_shuffle_dataset(x, y, percentage):
    x_noisy, y_noisy = shuffle(x, y)
    np.random.shuffle(y_noisy[:int(len(y_noisy) * percentage)])
    x_noisy, y_noisy = shuffle(x_noisy, y_noisy)
    return x_noisy, y_noisy



