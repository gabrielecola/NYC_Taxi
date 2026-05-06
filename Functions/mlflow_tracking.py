import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from config import MLFLOW_EXPERIMENT


def _log_pred_plot(y_true, y_pred, title: str, fname: str, out_dir: str) -> None:
    """Save a True-vs-Predicted scatter plot and log it as an MLflow artifact."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    xy_min = min(y_true.min(), y_pred.min())
    xy_max = max(y_true.max(), y_pred.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    mlflow.log_artifact(path, artifact_path=out_dir)


def log_all_runs(
    results: dict,
    X_test_sc,
    y_test,
    experiment: str = MLFLOW_EXPERIMENT,
) -> None:
    """
    Log every model result (metrics, params, plot, serialised model) to MLflow.

    Parameters
    ----------
    results   : output of modelling.train_evaluate()
    X_test_sc : scaled test features (numpy array)
    y_test    : true target values
    experiment: MLflow experiment name
    """
    mlflow.set_experiment(experiment)
    run_ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = f'plots_{run_ts}'

    with mlflow.start_run(run_name=f'all_models_{run_ts}'):
        mlflow.set_tags({
            'run_ts':     run_ts,
            'n_features': X_test_sc.shape[1],
            'ensemble':   'VotingRegressor',
        })

        best_rmse = float('inf')
        best_name = None

        for name, result in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                m = result['metrics']
                mlflow.log_metrics(m)
                print(
                    f"[{name:>10}]  RMSE={m['rmse']:.4f} | "
                    f"MAE={m['mae']:.4f} | R²={m['r2']:.4f}"
                )

                try:
                    mlflow.log_params(result['model'].get_params())
                except Exception:
                    mlflow.log_param('model_class', result['model'].__class__.__name__)

                _log_pred_plot(
                    y_test, result['predictions'],
                    title=f"{name}: Pred vs True",
                    fname=f"{name}_pred_vs_true.png",
                    out_dir=plots_dir,
                )

                input_example = np.asarray(X_test_sc[:5])
                signature = infer_signature(
                    np.asarray(X_test_sc), np.asarray(result['predictions'])
                )
                mlflow.sklearn.log_model(
                    result['model'],
                    name=name,
                    input_example=input_example,
                    signature=signature,
                )

                if m['rmse'] < best_rmse:
                    best_rmse = m['rmse']
                    best_name = name

        # Surface the best model on the parent run for quick comparison
        mlflow.log_metrics({'best_rmse': best_rmse})
        mlflow.log_param('best_model', best_name)
