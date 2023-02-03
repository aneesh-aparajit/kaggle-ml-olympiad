TRAIN_DATA = '../kaggle/input/ml-olympiad-waterqualityprediction/train.csv'
TEST_DATA = '../kaggle/input/ml-olympiad-waterqualityprediction/test.csv'
SUBMISSION_DATA = '../kaggle/input/ml-olympiad-waterqualityprediction/sample_submission.csv'
FOLD_DATA_PATH = '../kaggle/input/train_folds.csv'
TARGET_COL = 'result'
DROP_COLS = ['id']

import numpy as np

def root_mean_square_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((np.log(pred + 1) - np.log(true + 1))**2))

