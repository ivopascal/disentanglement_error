class DisentanglingModel:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict_disentangling(self, X):
        pass

    def score(self, y_true, y_pred):
        # Score must be positively increasing and what the uncertainty should reflect
        # For classification this is typically accuracy
        # For regression this is typically 1-MAE (when using std.dev as uncertainty)
        pass

    @property
    def is_regression(self):
        return False