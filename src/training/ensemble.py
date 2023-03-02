import pandas as pd
import numpy as np
import glob
from sklearn import metrics

import warnings
warnings.simplefilter('ignore')

from scipy.optimize import fmin
from functools import partial


class OptimizeRMLSE:
    def __init__(self):
        self._coef = None
    
    def _rlmse(self, coef, X, y):
        x_coef = X * coef
        preds = np.sum(x_coef, axis=1)
        preds = np.clip(preds, 0, np.infty)
        loss = np.sqrt(metrics.mean_squared_log_error(y_true=y, y_pred=preds))
        return loss
    
    def fit(self, X, y):
        loss_fn = partial(self._rlmse, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        print(f'Initial Weights: {init_coef}')
        self.coef_ = fmin(loss_fn, init_coef, disp=True)
    
    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions


def run_training(pred_df, pred_cols, fold):
    train_df = pred_df[pred_df.kfold_x != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold_x == fold].reset_index(drop=True)
    
    x_train = train_df[pred_cols].values
    x_valid = valid_df[pred_cols].values
    
    print("#"*15)
    opt = OptimizeRMLSE()
    opt.fit(x_train, train_df.result_x.values)
    preds = opt.predict(x_valid)
    preds = np.clip(preds, 0, np.infty)
    
    rmlse = np.sqrt(metrics.mean_squared_log_error(y_pred=preds, y_true=valid_df.result_x.values))
    
    print(f"Fold #{fold}, RMLSE: {rmlse}")
    print(f"Weights: {opt.coef_}")

    return opt.coef_


if __name__ == '__main__':
    paths = glob.glob('../preds/*.csv')
    print(paths)

    df = None
    for path in paths:
        if df is None:
            df = pd.read_csv(path)
        else:
            df = df.merge(pd.read_csv(path), on='id', how="left")

    print(df.shape)
    df = df.loc[:,~df.columns.duplicated()]
    print(df.shape)
    
    meta_cols = ['id']
    pred_cols = [x for x in df.columns if "preds" in x]
    

    weights = []
    for fold in range(5):
        weights.append(run_training(pred_df=df, pred_cols=pred_cols, fold=fold))
    
    print(weights)
