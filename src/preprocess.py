import pandas as pd
import os


def load_and_preprocess():
    df = pd.read_csv('data/AirQualityUCI.csv', sep=';', decimal=',')
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)
    df = df[['T', 'CO(GT)']].copy()
    df = df[(df['T'] != -200) & (df['CO(GT)'] != -200)]
    df = df.dropna()
    df = df.reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/processed.csv', index=False)

    msg = (
        "Preprocessing complete. "
        f"{len(df)} records saved to data/processed.csv"
    )
    print(msg)

    return df


if __name__ == '__main__':
    load_and_preprocess()
