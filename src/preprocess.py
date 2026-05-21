# preprocess.py
# Data Acquisition and Preprocessing Stage
# This script loads the Air Quality dataset, cleans it,
# and prepares it for model training.
# Reference: UCI Air Quality Dataset
# https://archive.ics.uci.edu/dataset/360/air+quality

import pandas as pd
import numpy as np
import os

def load_and_preprocess():
    df = pd.read_csv('data/AirQuality.csv', sep=';', decimal=',')
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)
    df = df[['T', 'CO(GT)']].copy()
    df = df[(df['T'] != -200) & (df['CO(GT)'] != -200)]
    df = df.dropna()
    df = df.reset_index(drop=True)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/processed.csv', index=False)
    print(f'Preprocessing complete. {len(df)} records saved to data/processed.csv')
    return df

if __name__ == '__main__':
    load_and_preprocess()
