import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

file_path = "data/final_dataset.csv"  
df = pd.read_csv(file_path)

df['Published'] = pd.to_datetime(df['Published'])

df_selected = df[['Published', 'Open', 'High', 'Low', 'Close']]

train_df = df_selected[df_selected['Published'] < '2025-01-01']
test_df = df_selected[df_selected['Published'] >= '2025-01-01']

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[['Open', 'High', 'Low', 'Close']])
test_scaled = scaler.transform(test_df[['Open', 'High', 'Low', 'Close']])

def create_sequences_multivariate(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1]) 
        y.append(data[i + sequence_length, -1])   
    return np.array(X), np.array(y)

def create_sequences_univariate(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X_train_all, y_train_all = create_sequences_multivariate(train_scaled, sequence_length)
X_test_all, y_test_all = create_sequences_multivariate(test_scaled, sequence_length)

train_close_scaled = scaler.fit_transform(train_df[['Close']])
test_close_scaled = scaler.transform(test_df[['Close']])
X_train_limited, y_train_limited = create_sequences_univariate(train_close_scaled, sequence_length)
X_test_limited, y_test_limited = create_sequences_univariate(test_close_scaled, sequence_length)

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error, r2_score

import os  

output_dir = "data/plots"
os.makedirs(output_dir, exist_ok=True)

def run_rnn(activation, optimizer, title_suffix):
    model_all = Sequential([
        SimpleRNN(50, activation=activation, input_shape=(sequence_length, X_train_all.shape[2])),
        Dense(1)
    ])
    model_all.compile(optimizer=optimizer, loss='mean_squared_error')
    model_all.fit(X_train_all, y_train_all, epochs=20, batch_size=32, verbose=0)
    predicted_close_scaled_all = model_all.predict(X_test_all)
    predicted_close_all = scaler.inverse_transform(
        np.hstack([np.zeros((len(predicted_close_scaled_all), 3)), predicted_close_scaled_all])
    )[:, -1]

    mse_all = mean_squared_error(
        scaler.inverse_transform(test_close_scaled[sequence_length:])[:, -1],
        predicted_close_all
    )
    r2_all = r2_score(
        scaler.inverse_transform(test_close_scaled[sequence_length:])[:, -1],
        predicted_close_all
    )

    model_limited = Sequential([
        SimpleRNN(50, activation=activation, input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model_limited.compile(optimizer=optimizer, loss='mean_squared_error')
    model_limited.fit(X_train_limited, y_train_limited, epochs=20, batch_size=32, verbose=0)
    predicted_close_scaled_limited = model_limited.predict(X_test_limited)
    predicted_close_limited = scaler.inverse_transform(predicted_close_scaled_limited)

    mse_limited = mean_squared_error(
        scaler.inverse_transform(test_close_scaled[sequence_length:])[:, -1],
        predicted_close_limited.flatten()
    )
    r2_limited = r2_score(
        scaler.inverse_transform(test_close_scaled[sequence_length:])[:, -1],
        predicted_close_limited.flatten()
    )
    plt.figure(figsize=(12, 8))
    plt.plot(test_df['Published'][sequence_length:], scaler.inverse_transform(test_close_scaled[sequence_length:]), label='Original Close Price', color='blue')
    plt.plot(test_df['Published'][sequence_length:], predicted_close_all, label=f'Regular (MSE={mse_all:.4f}, R²={r2_all:.4f})', color='green')
    plt.plot(test_df['Published'][sequence_length:], predicted_close_limited, label=f'With Sentiment (MSE={mse_limited:.4f}, R²={r2_limited:.4f})', color='orange')
    plt.xlabel('Published Date')
    plt.ylabel('Close Price')
    plt.title(f'Comparison of Actual vs Predicted Close Prices for RNN ({title_suffix})')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(output_dir, f"RNN_{title_suffix.replace(': ', '_').replace(', ', '_').replace(' ', '_')}.png")
    plt.savefig(plot_filename)
    plt.show()

configurations = [
    ('relu', 'adam', 'Activation: ReLU, Optimizer: Adam'),
    ('tanh', 'adam', 'Activation: Tanh, Optimizer: Adam'),
    ('sigmoid', 'adam', 'Activation: Sigmoid, Optimizer: Adam'),
    ('tanh', 'rmsprop', 'Activation: Tanh, Optimizer: RMSprop'),
    ('relu', 'adagrad', 'Activation: ReLU, Optimizer: Adagrad')
]

for activation, optimizer, title_suffix in configurations:
    run_rnn(activation, optimizer, title_suffix)