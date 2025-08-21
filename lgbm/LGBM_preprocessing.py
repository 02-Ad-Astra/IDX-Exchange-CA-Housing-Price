import pandas as pd
import numpy as np

NUMERIC_FEATURES = ['LivingArea', 'LotSizeAcres', 'HomeAge']
CATEGORICAL_FEATURES = ['PropertyType', 'City', 'PostalCode', 'LargeLot']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def preprocess_input(df, ref_values):
    df = df.copy()

    # Feature engineering
    if 'YearBuilt' in df.columns:
        df['HomeAge'] = 2025 - df['YearBuilt']
    else:
        df['HomeAge'] = ref_values['HomeAge']

    df['LargeLot'] = (df['LotSizeAcres'] > 0.5).astype("category")

    # Fill missing numeric values
    df['LivingArea'] = df['LivingArea'].fillna(ref_values['LivingArea'])
    df['LotSizeAcres'] = df['LotSizeAcres'].fillna(ref_values['LotSizeAcres'])
    df['HomeAge'] = df['HomeAge'].fillna(ref_values['HomeAge'])

    # Fill missing categorical values
    df['City'] = df['City'].fillna(ref_values['City']).astype('category')
    df['PostalCode'] = df['PostalCode'].fillna(ref_values['PostalCode']).astype('category')
    df['PropertyType'] = df['PropertyType'].fillna(ref_values['PropertyType']).astype('category')

    # Add 'Unknown' category before filling
    df['LargeLot'] = df['LargeLot'].astype('category')
    if 'Unknown' not in df['LargeLot'].cat.categories:
        df['LargeLot'] = df['LargeLot'].cat.add_categories(['Unknown'])
    df['LargeLot'] = df['LargeLot'].fillna('Unknown')

    return df[ALL_FEATURES]
