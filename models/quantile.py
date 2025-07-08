from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import numpy as np
from typing import List, Optional, Dict

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
    LightGBM‑based Quantile Gradient Boosting.

    ▸ Trains one LightGBM model per requested quantile.  
    ▸ Allows custom LightGBM hyper‑parameters via *lgbm_params*.  
    ▸ Passes *sample_weight* through to LightGBM to respect inverse‑probability
      weights when available.  
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        lgbm_params: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        quantiles : list[float]
            List of quantile levels (between 0 and 1).  Defaults to [0.1, 0.9].
        n_estimators : int
            Number of boosting iterations.
        learning_rate : float
            Shrinkage rate.
        lgbm_params : dict
            Additional parameters forwarded to `lgb.LGBMRegressor`.
            A sensible set of defaults is supplied when None.
        """
        if quantiles is None:
            quantiles = [0.1, 0.9]

        default_params = {
            # core
            "objective": "quantile",
            "num_leaves": 31,
            "min_child_samples": 5,
            "min_split_gain": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,          # suppress LightGBM training warnings
        }
        if lgbm_params is not None:
            default_params.update(lgbm_params)

        self.quantiles = quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        for q in quantiles:
            self.models[q] = lgb.LGBMRegressor(
                alpha=q,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                **default_params,
            )

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        """
        Fit a separate LightGBM model for each quantile.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
        y : array‑like, shape (n_samples,)
        sample_weight : array‑like, optional
            Observation‑level weights (e.g. from inverse‑probability weighting).
        """
        for model in self.models.values():
            model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X, quantiles: Optional[List[float]] = None) -> np.ndarray:
        """
        Predict conditional quantiles.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
        quantiles : list[float], optional
            Which quantiles to return.  Defaults to all fitted levels.

        Returns
        -------
        np.ndarray, shape (n_samples, len(quantiles))
        """
        if quantiles is None:
            quantiles = self.quantiles

        preds = []
        for q in quantiles:
            try:
                preds.append(self.models[q].predict(X))
            except KeyError as err:
                raise ValueError(
                    f"Quantile {q} requested but model was not fitted for it."
                ) from err
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