import pandas as pd
import pymssql
from matplotlib import pyplot as plt
import numpy as np
import mplfinance as mpf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import yaml
import joblib


def add_months(date_str, month_num):
    
    current_year = int(date_str[:4])
    current_month = int(date_str[4:6])
    new_month = (current_month + month_num)
    new_year = current_year
    # should be constrained within [1, 12]
    if new_month > 12:
        new_month = new_month - 12
        new_year += 1

    if new_month < 10:
        new_month = '0' + str(new_month)
    else:
        new_month = str(new_month)
    
    return str(new_year) + new_month + date_str[-2:]
    
def save_result(result_df):
    
    result_df.to_csv(f"rolling_train_temp.csv")

    print("Result Saved Correctly")  


def filter_first_occurrences(df, keep='last'):
    
    # Sort the DataFrame by 'date' and 'code' to ensure the first occurrence is at the top
    df_sorted = df.sort_values(by=['date', 'code'])
    
    # Drop duplicates based on 'date' and 'code', keeping the first occurrence
    df_filtered = df_sorted.drop_duplicates(subset=['date', 'code'], keep=keep)
    
    return df_filtered[list(df.columns)].dropna()



def add_noise_4d(X, noise_level=1e-6):
    """
    Add a small amount of noise to the 4D data in X to prevent numerical issues.
    
    """
    # Generate noise with the same shape as X
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    # Add noise to the data
    X_with_noise = X + noise
    
    return X_with_noise


def standardize_training_data(df, feature_list):
    """
    Standardize the training dataframe and filter the dirty data.
    Returns the standardized dataframe and a dictionary of scalers.
    """
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    scalers = {}
    
    for feature in feature_list:
        if feature in df.columns:
            series = df[feature].copy()
            series_reshaped = np.array(series.values).reshape(-1, 1)
            
            scaler = StandardScaler()
            series_scaled = scaler.fit_transform(series_reshaped)
            df[feature] = series_scaled.flatten()
            
            # Save the scaler for this feature
            scalers[feature] = scaler
        else:
            print(f"Feature '{feature}' not found in DataFrame columns")
    
    return df[feature_list], scalers


def standardize_testing_data(df, feature_list, scalers):
    """
    Standardize the testing dataframe using the scalers fitted on training data.
    """
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    
    for feature in feature_list:
        if feature in df.columns:
            if feature in scalers:
                series = df[feature].copy()
                series_reshaped = series.values.reshape(-1, 1)
                series_scaled = scalers[feature].transform(series_reshaped)
                
                # back to the original dataframe
                df[feature] = series_scaled.flatten()
            else:
                print(f"No scaler found for feature '{feature}'")
        else:
            print(f"Feature '{feature}' not found in DataFrame columns")
    
    return df[feature_list]


def Creat_Sequence(stock_data, feature_list, seq_length=5, predict_length=7):
    sequences = []
    labels = []

    df = stock_data.copy()
    label_seies = df['Label'].shift(predict_length).fillna(0).copy()

    for i in range(len(df) - seq_length - predict_length):
        sequence = df[feature_list].iloc[i:i+seq_length, :].values.tolist()
        label = label_seies.iloc[i+seq_length]
        sequences.append(sequence)
        labels.append(label)

    # Convert sequences and labels to NumPy arrays
    sequences = np.array(sequences)  
    labels = np.array(labels)

    return sequences, labels


import torch
from sklearn.preprocessing import minmax_scale
def buying_index(reg_model, X_for_predict, predict_length):
    """
    This function predicts future closing prices and identifies potential buying opportunities.

    Args:
        reg_model: Trained regression model (GRUModel in your case)
        X_for_predict: A list or NumPy array containing the input sequence for prediction.
        predict_length: The number of days to predict into the future.

    Returns:
        None (Displays a plot with past and predicted closing prices and a potential buying index).
    """

    # Convert input to PyTorch tensor
    X_tensor = torch.tensor(X_for_predict, dtype=torch.float)

    # Predict closing prices
    y_predict = reg_model.model(X_tensor)
    y_predict_numpy = y_predict.detach().numpy()

    y_scaled = minmax_scale(y_predict_numpy)
    y_scaled = y_predict_numpy
    # Separate past and predicted prices 
    past_prices = y_scaled[-2*predict_length:-predict_length]
    future_prices = y_scaled[-predict_length:]

    return y_scaled, past_prices, future_prices