from sklearn.metrics import accuracy_score, mean_absolute_error


class DisentanglingModel:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict_disentangling(self, X):
        pass

    def score(self, y_true, y_pred):
        # Score must be positively increasing and what the uncertainty should reflect
        if self.is_regression:
            return 1 - mean_absolute_error(y_true, y_pred)
        else:
            return accuracy_score(y_true, y_pred)

    @property
    def is_regression(self):
        return False