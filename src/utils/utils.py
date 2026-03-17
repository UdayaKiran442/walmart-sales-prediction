import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split_date(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = (df.Date.dt.isocalendar().week)*1.0  
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(['Date', 'Temperature','Fuel_Price', 'Type', 'MarkDown1', 'MarkDown2', 'MarkDown3',
             'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment'], axis=1)

def binary_to_int(X):
    """Convert binary columns to int type"""
    return X.astype(int)

def get_transformation_pipeline(binary_cols: list, numeric_cols: list) -> Pipeline:
    # ft = FunctionTransformer(lambda x: x.astype(int))
    ft = FunctionTransformer(binary_to_int)
    sc = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', ft, binary_cols),
            ('numeric', sc, numeric_cols)
        ]
    )
    pipeline = Pipeline([('preprocessor', preprocessor)])
    return pipeline


def save_numpy_array_data(file_path: str, array: np.array):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
        np.save(file_obj, array)
    
def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        print(f"Error occurred while saving object using pickle: {e}")
        raise e
