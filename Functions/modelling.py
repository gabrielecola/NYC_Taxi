import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import MODEL_PARAMS

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def build_models(use_xgb: bool = True) -> list:
    """
    Return a list of (name, estimator) pairs.
    Appends XGBoost if available and use_xgb=True, else LightGBM.
    """
    p = MODEL_PARAMS
    models = [
        ('lr',    LinearRegression()),
        ('ridge', Ridge(**p['ridge'])),
        ('lasso', Lasso(**p['lasso'])),
        ('rf',    RandomForestRegressor(**p['rf'])),
        ('gb',    GradientBoostingRegressor(**p['gb'])),
        ('mlp',   MLPRegressor(**p['mlp'])),
    ]
    if use_xgb and HAS_XGB:
        models.append(('xgb', XGBRegressor(**p['xgb'])))
    elif HAS_LGBM:
        models.append(('lgbm', LGBMRegressor(**p['lgbm'])))
    return models


def get_metrics(y_true, y_pred) -> dict:
    """Return RMSE, MAE and R² as a dict."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae':  mean_absolute_error(y_true, y_pred),
        'r2':   r2_score(y_true, y_pred),
    }


def train_evaluate(models: list, X_train_sc, X_test_sc, y_train, y_test) -> dict:
    """
    Fit each base model, then fit a VotingRegressor on top.

    Returns
    -------
    dict keyed by model name, each value is:
        {'model': fitted_estimator, 'predictions': array, 'metrics': dict}
    """
    results = {}

    for name, model in models:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        results[name] = {
            'model':       model,
            'predictions': y_pred,
            'metrics':     get_metrics(y_test, y_pred),
        }

    # Ensemble: reuse already-fitted estimators
    voting_reg = VotingRegressor(
        estimators=[(name, res['model']) for name, res in results.items()]
    )
    voting_reg.fit(X_train_sc, y_train)
    y_pred_ens = voting_reg.predict(X_test_sc)

    results['voting'] = {
        'model':       voting_reg,
        'predictions': y_pred_ens,
        'metrics':     get_metrics(y_test, y_pred_ens),
    }

    return results
