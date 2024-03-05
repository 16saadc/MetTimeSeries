import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

import streamlit as st

def plot_data_distributions(data):
    n = len(data.columns)
    ncols = 3 
    nrows = n // ncols + (n % ncols > 0) 
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    
    axes = axes.flatten()
    
    for i, column in enumerate(data.columns):
        sns.histplot(data[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
    
    for i in range(n, nrows * ncols):
        fig.delaxes(axes[i])
    
    st.pyplot(fig)
   
def plot_correlation_matrix(data):
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax)
    
    ax.set_title('Correlation Matrix')
    
    st.pyplot(fig) 

def plot_feature_boxplots(data):
    num_columns = data.shape[1]
    num_rows = num_columns // 3 + (num_columns % 3 > 0)
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 4))

    if num_columns > 1:
        axes = axes.flatten()
    
    for i, column in enumerate(data.columns):
        sns.boxplot(y=data[column], ax=axes[i])
        axes[i].set_title(f'Boxplot for {column}')
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-tick labels

    for j in range(i+1, num_rows * 3):
        fig.delaxes(axes[j])

    fig.tight_layout()
    st.pyplot(fig)


@st.cache_data
def preprocess_data(data, features_to_drop, features_to_scale, target_column='X'):
    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('drop', 'drop', features_to_drop),
            ('std_scale', MinMaxScaler(), features_to_scale),
        ],
        remainder='passthrough'
    )

    # Create the preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Apply the pipeline to the data
    X = data.drop(columns=[target_column])
    y = data[target_column].values
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed, y


@st.cache_data
def remove_outliers(X_preprocessed, y):
    # Assuming X_preprocessed is a NumPy array or a pandas DataFrame
    # Calculate Q1 and Q3
    Q1 = np.percentile(X_preprocessed, 25, axis=0)
    Q3 = np.percentile(X_preprocessed, 75, axis=0)
    # Calculate the IQR
    IQR = Q3 - Q1

    # Define the outlier step (1.5 times IQR)
    outlier_step = 1.5 * IQR

    # Determine a list of indices of outliers for features
    outlier_list_col = []
    for feature_idx in range(X_preprocessed.shape[1]):
        # Find indices of outliers for column feature_idx
        outlier_list_col.extend(np.where((X_preprocessed[:, feature_idx] < Q1[feature_idx] - outlier_step[feature_idx]) |
                                        (X_preprocessed[:, feature_idx] > Q3[feature_idx] + outlier_step[feature_idx]))[0])

    # Select observations without outliers
    outlier_indices = np.unique(outlier_list_col)
    X_preprocessed = np.delete(X_preprocessed, outlier_indices, axis=0)
    y = np.delete(y, outlier_indices, axis=0)

    return X_preprocessed, y

@st.cache_data
def create_sequences(inputs, targets, time_steps=1):
    input_seqs, target_seqs = [], []
    for i in range(len(inputs) - time_steps):
        input_seqs.append(inputs[i:(i + time_steps), :])
        target_seqs.append(targets[i + time_steps])
    return np.array(input_seqs), np.array(target_seqs)

def plot_training_validation_curve(history):
    # Extract the history data
    training_loss = history.history['loss']
    validation_loss = history.history.get('val_loss', None)

    # Determine the number of epochs
    epochs = range(1, len(training_loss) + 1)

    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss
    ax.plot(epochs, training_loss, 'bo-', label='Training loss')
    if validation_loss:
        ax.plot(epochs, validation_loss, 'ro-', label='Validation loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

@st.cache_data
def split_data(X, y, time_steps):
    # Sequence creation and data splitting (assumes these functions are defined)
    X, y = create_sequences(X, y, time_steps)
    st.write("Split the model into Train, Val, test sets (70%, 15%, 15%)")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

@st.cache_data
@st.cache_resource
def train_model(X_train, X_val, y_train, y_val, epochs, learning_rate, time_steps):

    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1),
    ])

    st.write(model.summary())
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs,
        validation_data=(X_val, y_val), 
        verbose=1)
    
    return history, model

@st.cache_data
def run_tests(_model, y_test, X_test):
    y_pred = _model.predict(X_test)

    results = get_metrics(y_pred, y_test)
    st.table(results)

    plot_actual_vs_predicted(y_test, y_pred)

    plot_residuals(y_test, y_pred)

@st.cache_data
def get_metrics(y_pred, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(y_test, label='Actual')
    ax.plot(y_pred, label='Predicted')
    ax.set_title('Actual vs Predicted Values')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Target Variable')
    ax.legend()
    st.pyplot(fig)


def plot_residuals(y_test, y_pred):

    fig, ax = plt.subplots(figsize=(12,6))

    residuals = y_test - y_pred.flatten()

    ax.plot(residuals)
    ax.set_title('Residuals of Predictions')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Residual')
    ax.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig)

    # Assuming 'residuals' are already computed as y_test - y_pred
    dw_statistic = durbin_watson(residuals)
    st.write(f'Durbin-Watson Statistic: {dw_statistic}')


    # Plot the autocorrelation of the residuals
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_acf(residuals, lags=50, alpha=0.05, ax=ax)
    ax.set_title('Autocorrelation Function')
    st.pyplot(fig)
