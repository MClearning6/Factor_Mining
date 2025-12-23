# src/data/data_check.py
import pandas as pd

required_col = {'date','asset'}

def check_df(df):
    missing = required_col - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise TypeError('date column must be datetime')

    df = df.sort_values(['asset', 'date']).reset_index(drop=True)
    
    print('Yo, your data is fucking awesome bro')

    return df