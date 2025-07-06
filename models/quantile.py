from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import numpy as np

class RandomForestRegressorWrapper:
    """
    Wrapper for standard Random Forest Regressor.
    
    Predicts conditional mean of the response variable.
    """
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        """
        Fit the random forest model.
        
        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the mean outcome for the input features.
        
        Returns:
        - np.ndarray: Predicted means with shape (n_samples,).
        """
        return self.model.predict(X)

class QuantileGradientBoosting:
    """
    Quantile Gradient Boosting using LightGBM.
    
    Trains separate models for each quantile.
    """
    def __init__(self, quantiles=[0.1, 0.9], n_estimators=100, **kwargs):
        self.models = {}
        for q in quantiles:
            model = lgb.LGBMRegressor(objective='quantile', alpha=q,
                                      n_estimators=n_estimators, **kwargs)
            self.models[q] = model
        self.quantiles = quantiles

    def fit(self, X, y):
        """
        Fit quantile gradient boosting models.
        
        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.
        """
        for model in self.models.values():
            model.fit(X, y)

    def predict(self, X, quantiles=None):
        """
        Predict specified quantiles using LightGBM models.

        Parameters:
        - X: Feature matrix
        - quantiles: Optional[List[float]] (defaults to self.quantiles)

        Returns:
        - np.ndarray: shape (n_samples, len(quantiles))
        """
        if quantiles is None:
            quantiles = self.quantiles

        preds = []
        for q in quantiles:
            if q in self.models:
                preds.append(self.models[q].predict(X))
            else:
                raise ValueError(f"Quantile {q} not available in trained models.")
        return np.stack(preds, axis=1)

class GradientBoostingRegressorWrapper:
    """
    Wrapper for standard Gradient Boosting Regressor.
    
    Predicts conditional mean of the response variable.
    """
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(**kwargs)

    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the mean outcome for the input features.
        
        Returns:
        - np.ndarray: Predicted means with shape (n_samples,).
        """
        return self.model.predict(X)