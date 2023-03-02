import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
from xgboost import XGBRegressor

from scipy.optimize import fmin
from functools import partial

def rmsle(predicted, actual):
    '''https://www.kaggle.com/code/bachaboos/rmsle'''
    predicted = np.array(predicted) + 1
    actual = np.array(actual) + 1

    # Calculate the logarithmic values
    predicted_log = np.log(predicted)
    actual_log = np.log(actual)

    # Calculate the difference between the predicted and actual logarithmic values
    diff = predicted_log - actual_log

    # Square the differences
    diff_squared = diff ** 2

    # Calculate the mean of the squared differences
    mean_diff_squared = np.mean(diff_squared)

    # Take the square root of the mean of the squared differences
    return np.sqrt(mean_diff_squared)


class OptimizeRMLSE:
    def __init__(self):
        self.coef_ = None
    
    def _rmlse(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        rmlse_score = rmsle(predicted=predictions, actual=y)
        return rmlse_score

    def fit(self, X, y):
        partial_loss = partial(self._rmlse, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        print(f'Initial Weights: {init_coef}')
        self.coef_ = fmin(partial_loss, init_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions


MODELS = sorted([x+'_preds' for x in ['cat', 'kNN', 'lasso', 'lgbm', 'lr', 'rf', 'ridge', 'svm', 'xgb']])
print(MODELS)

df = None



def run_training(fold):
    train_df = df[df.kfold != fold]
    valid_df = df[df.kfold == fold]
    
    valid_idx = df[df.kfold == 0].index.to_list()
    
    xtrain, ytrain = train_df.drop(['kfold', 'result'], axis=1).values, train_df.result.values
    xvalid, yvalid = valid_df.drop(['kfold', 'result'], axis=1).values, valid_df.result.values

    print('#'*15)
    print(f'### Fold #{fold}')
    print('#'*15)
    
    model = OptimizeRMLSE()
    model.fit(xtrain, ytrain)
    
    valid_pred = model.predict(xvalid)
    train_pred = model.predict(xtrain)
    
    train_rmlse = rmsle(predicted=train_pred, actual=ytrain)
    valid_rmlse = rmsle(predicted=valid_pred, actual=yvalid)
    
    print(f'Train RMLSE: {train_rmlse}')
    print(f'Valid RMLSE: {valid_rmlse}')
    
    # pred_df.loc[valid_idx, f"preds"] = valid_pred

    return model, train_rmlse, valid_rmlse


def create_df(paths):
    dfs = []
    for ix, path in enumerate(paths):
        df = pd.read_csv(path)
        dfs.append(df.loc[:, f'{MODELS[ix]}'])
    result = pd.read_csv(paths[0]).loc[:, 'result']
    dfs.append(result)
    return pd.concat(dfs, axis=1).reset_index(drop=True)

if __name__ == '__main__':
    paths = sorted(glob('../preds/*.csv'))
    df = create_df(paths)
    
    df.clip(0, np.infty, inplace=True)

    for col in MODELS:
        print(f'# {col} RMLSE: {rmsle(predicted=df[col], actual=df.result)}')

    df.loc[:, "kfold"] = -1
    X = df.drop(['result', 'kfold'], axis=1)
    y = df.result

    kf = model_selection.KFold()
    for fold, (train, valid) in enumerate(kf.split(X, y)):
        df.loc[valid, "kfold"] = fold

    for fold in range(5):
        model, train_rmlse, valid_rmlse = run_training(fold)

    print('Simple Average: ')
    print(rmsle(predicted=df[MODELS].mean(axis=1), actual=df['result']))
