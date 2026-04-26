"""Model training and evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(model, X, y_log, label=''):
    """Return a dict of metrics on log-price scale and back-transformed."""
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_log)
    return {
        'label': label,
        'rmse_log': rmse(y_log, y_pred_log),
        'mae_usd': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_log, y_pred_log),
        'mape': mape(y_true, y_pred),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    return (
        pd.DataFrame(results)
        .set_index('label')
        .sort_values('rmse_log')
        .style.format({'rmse_log': '{:.4f}', 'mae_usd': '${:,.0f}',
                       'r2': '{:.4f}', 'mape': '{:.1f}%'})
        .background_gradient(subset=['rmse_log'], cmap='RdYlGn_r')
    )
