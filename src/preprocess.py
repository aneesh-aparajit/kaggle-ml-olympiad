import pandas as pd
import numpy as np
from sklearn import preprocessing
import config
from typing import List, Dict, Tuple, Optional


def build_mapping_dict_categorical(df: pd.DataFrame, categoric_cols: List[str]) -> Dict[str, List[str]]:
    categoric_dict = {}

    for column in categoric_cols:
        categoric_dict[column] = df[column].value_counts(:50).index.to_list()

    return categoric_dict


def map_mapping_dict_categoric(df: pd.DataFrame, categoric_dict: Dict[str, List[str]], categoric_cols: List[str]) -> pd.DataFrame:
    for column in categoric_cols:
        df[column] = df[column].apply(lambda x: x if x in categoric_dict[column] else 'unk')
    return df


def get_std_scaler_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> preprocessing.StandardScaler:
    scaler = StandardScalar()
    scaler.fit(df)
    return scalar


def get_minmax_scaler_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> preprocessing.MinMaxScaler:
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df)
    return scaler


if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_DATA)
    df = df.drop(config.DROP_COLS, axis=1)

    numerical_cols = df.select_dtypes(include=np.number).columns.to_list()
    categoric_cols = df.select_dtypes(include='object').columns.to_list()

    X = df.drop(config.TARGET_COL, axis=1)
    y = df[config.TARGET_COL]



