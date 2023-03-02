import numpy as np
import pandas as pd
from sklearn import metrics, neighbors
import copy
import os
import pickle


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


df = pd.read_csv('../training_folds.csv')

pred_df = copy.deepcopy(df)
pred_df["fold_preds"] = -1

def lr_run_training(fold):
    train_df = df[df.kfold != fold]
    valid_df = df[df.kfold == fold]
    
    valid_idx = df[df.kfold == 0].index.to_list()
    
    xtrain, ytrain = train_df.drop(['kfold', 'result'], axis=1).values, train_df.result.values
    xvalid, yvalid = valid_df.drop(['kfold', 'result'], axis=1).values, valid_df.result.values
    
    model = neighbors.KNeighborsRegressor()
    
    model.fit(xtrain, ytrain)
    
    valid_pred = model.predict(xvalid)
    train_pred = model.predict(xtrain)
    
    train_rmlse = rmsle(predicted=train_pred, actual=ytrain)
    valid_rmlse = rmsle(predicted=valid_pred, actual=yvalid)
    
    print('#'*15)
    print(f'### Fold #{fold}')
    print('#'*15)
    
    print(f'Train RMLSE: {train_rmlse}')
    print(f'Valid RMLSE: {valid_rmlse}')
    
    pred_df.loc[valid_idx, f"fold_preds"] = valid_pred

    return model, train_rmlse, valid_rmlse,


if __name__ == '__main__':
    models = []
    train_losses = []
    valid_losses = []

    for fold in range(5):
        model, train_loss, valid_loss = lr_run_training(fold)
        models.append(model)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print('\n')

    # print(pred_df.head())

    pred_df.to_csv('../preds/kNN_preds.csv')

    if os.path.exists('./model_ckpt/kNN/'):
        pass
    else:
        os.mkdir('./model_ckpt/kNN/')
    
    for fold, model in enumerate(models):
        pickle.dump(model, open(f'./model_ckpt/kNN/kNN_{fold}.bin', 'wb'))
    
