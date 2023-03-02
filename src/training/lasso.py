import pandas as pd
import numpy as np
from sklearn import metrics, linear_model
import pickle
import os
from best_params import LASSO


def run_training(fold):
    df = pd.read_csv('../../training_folds.csv')
    train_df = df[df.kfold != fold]
    valid_df = df[df.kfold == fold]
    preds_df = valid_df.copy()
    preds_df.loc[:, "lasso_preds"] = -1

    xtrain, ytrain = train_df.drop(['id', 'result', 'kfold'], axis=1).values, train_df.result.values
    xvalid, yvalid = valid_df.drop(['id', 'result', 'kfold'], axis=1).values, valid_df.result.values

    model = linear_model.Ridge(**LASSO)

    model.fit(xtrain, ytrain)
    preds = model.predict(xvalid)
    preds = np.clip(preds, 0, np.infty)

    print(f'Fold #{fold}\t Loss: {np.sqrt(metrics.mean_squared_log_error(y_true=yvalid, y_pred=preds))}')

    preds_df.loc[valid_df.index, "lasso_preds"] = preds

    return model, preds_df[["id", "result", "kfold", "lasso_preds"]]


if __name__ == '__main__':
    run_training(fold=0)

    dfs = []
    models = []

    for fold in range(5):
        model, df = run_training(fold)
        models.append(model)
        dfs.append(df)
    
    df = pd.concat(dfs).reset_index(drop=True)

    if os.path.exists('../models/lasso/'):
        pass
    else:
        os.mkdir('../models/lasso/')

    for fold, model in enumerate(models):
        pickle.dump(model, open(f'../models/lasso/lasso-{fold}.bin', 'wb'))
    
    print(df.shape)
    df.to_csv('../preds/lasso_preds.csv', index=False)

    print(df.head())
    print(np.sqrt(metrics.mean_squared_log_error(y_true=df['result'], y_pred=df['lasso_preds'])))
