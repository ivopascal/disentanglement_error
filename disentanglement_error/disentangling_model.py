from abc import abstractmethod, ABC

from sklearn.metrics import accuracy_score, mean_absolute_error


class DisentanglingModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict_disentangling(self, x_test):
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