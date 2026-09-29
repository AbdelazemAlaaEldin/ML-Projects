from sklearn.base import BaseEstimator, TransformerMixin


class MushroomPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Replace missing-like values
        X = X.replace("?", "Unknown")

        # Remove constant feature
        X = X.drop("veil-type", axis=1)

        return X