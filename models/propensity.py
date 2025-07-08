from sklearn.linear_model import LogisticRegression

def fit_propensity(X, T):
    """
    Fit a propensity score model using logistic regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The feature matrix.
    T : array-like, shape (n_samples,)
        The treatment assignment (binary: 0 or 1).

    Returns
    -------
    model : LogisticRegression
        The fitted logistic regression model.
    """

    model = LogisticRegression(solver='liblinear')
    model.fit(X, T)
    
    return model

def predict_propensity(X, model):
    """
    Predict propensity scores using the fitted model.

    Parameters
    ----------
    model : LogisticRegression
        The fitted logistic regression model.
    X : array-like, shape (n_samples, n_features)
        The feature matrix for which to predict propensity scores.

    Returns
    -------
    propensity_scores : array, shape (n_samples,)
        The predicted propensity scores.
    """
    
    propensity_scores = model.predict_proba(X)[:, 1]
    print("[DEBUG][DEBUG][DEBUG][DEBUG] Propensity scores (first 10):", propensity_scores[:10])
    
    return propensity_scores