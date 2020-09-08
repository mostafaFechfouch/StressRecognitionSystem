# This script do the data treatment and generate a single dataframe row in JSON format

import numpy as np
import pandas as pd
from scipy import signal
import joblib

knn = joblib.load("knn_model.pkl") 

dataframe_hrv = pd.read_csv("df.csv")
dataframe_hrv = dataframe_hrv.reset_index(drop=True)

def fix_stress_labels(df='',label_column='stress'):
    df['stress'] = np.where(df['stress']>=0.5, 1, 0)
    return df
dataframe_hrv = fix_stress_labels(df=dataframe_hrv)

def missing_values(df):
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13) 
    df=df.fillna(df.mean())
    return df

dataframe_hrv = missing_values(dataframe_hrv)

selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']
dataframe_hrv=dataframe_hrv[selected_x_columns]

# generating one row
row1 = dataframe_hrv.sample(n = 1) 

print(knn.predict(row1))

d=row1.to_json()

print(d)