"""Reusable preprocessing utilities for the used-car price prediction pipeline."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


PRICE_MIN = 500
PRICE_MAX = 150_000
YEAR_MIN = 1980
YEAR_MAX = 2025
ODO_MAX = 400_000
CURRENT_YEAR = 2025

CONDITION_ORDER = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']

CYLINDERS_MAP = {
    '3 cylinders': 3, '4 cylinders': 4, '5 cylinders': 5,
    '6 cylinders': 6, '8 cylinders': 8, '10 cylinders': 10,
    '12 cylinders': 12, 'other': 0,
}

USE_COLS = [
    'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders',
    'fuel', 'odometer', 'title_status', 'transmission', 'drive',
    'size', 'type', 'paint_color', 'state', 'lat', 'long'
]

CAT_COLS = [
    'manufacturer', 'fuel', 'title_status', 'transmission',
    'drive', 'type', 'paint_color', 'state'
]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=USE_COLS, low_memory=False)


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df['price'].between(PRICE_MIN, PRICE_MAX) &
        df['year'].between(YEAR_MIN, YEAR_MAX) &
        (df['odometer'].isna() | df['odometer'].between(0, ODO_MAX))
    )
    return df[mask].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vehicle_age'] = CURRENT_YEAR - df['year']
    df['log_odometer'] = np.log1p(df['odometer'])
    df['log_price'] = np.log1p(df['price'])
    df['cylinders_int'] = df['cylinders'].map(CYLINDERS_MAP)

    # Ordinal condition
    cond_map = {v: i for i, v in enumerate(CONDITION_ORDER)}
    df['condition_ord'] = df['condition'].map(cond_map)

    # State-level median price (computed on training set only — passed in)
    return df


def impute_categoricals(df: pd.DataFrame, fill_value: str = 'unknown') -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS + ['condition', 'size']:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    df['cylinders_int'] = df['cylinders_int'].fillna(0)
    df['condition_ord'] = df['condition_ord'].fillna(-1)
    df['log_odometer'] = df['log_odometer'].fillna(df['log_odometer'].median())
    return df
