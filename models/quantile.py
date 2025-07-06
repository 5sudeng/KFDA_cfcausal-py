from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from bartpy.sklearnmodel import SklearnModel
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
        self.models = []
        for q in quantiles:
            model = lgb.LGBMRegressor(objective='quantile', alpha=q,
                                      n_estimators=n_estimators, **kwargs)
            self.models.append(model)
        self.quantiles = quantiles

    def fit(self, X, y):
        """
        Fit quantile gradient boosting models.
        
        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.
        """
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        """
        Predict quantiles using fitted LightGBM models.
        
        Returns:
        - np.ndarray: Predicted quantiles with shape (n_samples, n_quantiles).
        """
        preds = [model.predict(X) for model in self.models]
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

class QuantileBART:
    """
    Quantile estimation using Bayesian Additive Regression Trees (BART).
    
    Predicts posterior quantiles from sampled posterior predictions.
    """
    def __init__(self, quantiles=[0.1, 0.9], **kwargs):
        self.quantiles = quantiles
        self.model = SklearnModel(**kwargs)

    def fit(self, X, y):
        """
        Fit the BART model on training data.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict posterior quantiles for input features.

        Returns:
        - np.ndarray: Quantile predictions with shape (n_samples, n_quantiles).
        """
        posterior_samples = self.model.predict(X, return_type='samples')
        return np.quantile(posterior_samples, self.quantiles, axis=0).T

class BARTRegressor:
    """
    Standard regression using Bayesian Additive Regression Trees (BART).
    
    Predicts the posterior mean.
    """
    def __init__(self, **kwargs):
        self.model = SklearnModel(**kwargs)

    def fit(self, X, y):
        """
        Fit the BART model on training data.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict posterior mean for input features.

        Returns:
        - np.ndarray: Mean predictions with shape (n_samples,).
        """
        return self.model.predict(X)