import pickle
import numpy as np
import pandas as pd

def fill_missing_values(df, df_mean, col2use):
    for c in col2use:
        assert c in df.columns, c + ' not in df'
        assert c in df_mean.col.values, c+ 'not in df_mean'
    
    for c in col2use:
        mean_value = df_mean.loc[df_mean.col == c,'mean_val'].values[0]
        df[c] = df[c].fillna(mean_value)
    return df

def load_data(model_path, input_cols_path, mean_path, scalar_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(input_cols_path, 'rb') as f:
        input_cols = pickle.load(f)
    with open(mean_path, 'rb') as f:
        mean = pd.read_csv(f, names=['col', 'mean_val'])
    with open(scalar_path, 'rb') as f:
        scalar = pickle.load(f)
    return model, input_cols, mean, scalar

def create_dataframe(data, input_cols):
    df = pd.DataFrame(data)
        
    num_col = df.select_dtypes(include=['int64', 'float64']).columns
    cat_col = df.select_dtypes(include=['object']).columns
    
    num_col = num_col.drop('duration', errors='ignore')  
    
    new_cat_col = pd.get_dummies(df[cat_col], drop_first=False)
    all_cat_col = [col for col in input_cols if col not in num_col]  
    
    new_cat_col = new_cat_col.reindex(columns=all_cat_col, fill_value=0)
    
    input_columns = num_col.tolist() + all_cat_col
    new_df = pd.concat([df[num_col], new_cat_col], axis=1)
    
    new_df = new_df[input_columns]
    
    return new_df


def preprocess_data(df, input_cols, mean, scalar):
    
    df = fill_missing_values(df, mean, input_cols)
    
    X = df[input_cols].values
    
    X_scaled = scalar.transform(X)
    
    return X_scaled

def predict_y(X, model):
    return model.predict_proba(X)[:, 1]
