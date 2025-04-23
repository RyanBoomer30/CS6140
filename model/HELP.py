import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import time

# Constants
SEQ_LEN = 3
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y = [], []
        for i in range(len(X) - seq_len):
            if i + seq_len < len(X) and i + seq_len < len(y):
                self.X.append(X[i:i + seq_len])
                self.y.append(y[i + seq_len])
        if not self.X:  
            print(f"Warning: Not enough data to create sequences of length {seq_len}. Using smaller sequence length.")
            seq_len = min(len(X) // 2, len(y) // 2) 
            for i in range(len(X) - seq_len):
                self.X.append(X[i:i + seq_len])
                self.y.append(y[i + seq_len])
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        print(f"Created dataset with {len(self.X)} sequences of length {seq_len}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class CustomRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class CustomLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_model(model, train_loader, epochs=EPOCHS, lr=0.001, optimizer_type='adam'):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer_type == 'adam' \
        else torch.optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch).view(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses


def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(DEVICE)
        preds = model(X_test_tensor).view(-1).cpu().numpy()
    y_true = y_test_tensor.numpy()
    return preds, y_true, mean_squared_error(y_true, preds), r2_score(y_true, preds)


def plot_predictions(results, title="Prediction Comparison"):
    """
    Enhanced visualization comparing with/without sentiment predictions
    
    Args:
        results: Dictionary containing prediction results
    """
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models), sharex=True)
    
    if n_models == 1:
        axes = [axes]
    
    # Use distinctive color palette
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, model_results) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot actual values
        ax.plot(model_results['true'], color=colors[0], label='Actual', linewidth=2)
        
        # Plot predictions
        for j, (pred_type, pred_data) in enumerate(model_results['predictions'].items()):
            ax.plot(pred_data, color=colors[j+1], label=pred_type, linewidth=1.5, 
                   linestyle='--' if 'with sentiment' in pred_type else '-.')
        
        # Add metrics to legend
        for pred_type, metrics in model_results['metrics'].items():
            ax.plot([], [], ' ', label=f"{pred_type}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
        
        ax.set_title(f"{model_name} Predictions", fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add y-label to the leftmost subplot
        ax.set_ylabel('Normalized Price', fontsize=12)
    
    # Add x-label to the bottom subplot
    axes[-1].set_xlabel('Time Steps', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()


def plot_comparison_metrics(results, title="Impact of Sentiment Data on Model Performance"):
    models = []
    mse_without = []
    mse_with = []
    r2_without = []
    r2_with = []
    
    for model_type in ['RNN', 'LSTM']:
        for i in range(1, 6):
            model_name = f"{model_type}_{i}"
            if model_name in results['without_sentiment'] and model_name in results['with_sentiment']:
                models.append(f"{model_type} {i}")
                mse_without.append(results['without_sentiment'][model_name]['metrics']['mse'])
                mse_with.append(results['with_sentiment'][model_name]['metrics']['mse'])
                r2_without.append(results['without_sentiment'][model_name]['metrics']['r2'])
                r2_with.append(results['with_sentiment'][model_name]['metrics']['r2'])
    
    mse_change = [(mse_with[i] - mse_without[i]) / mse_without[i] * 100 for i in range(len(models))]
    r2_change = [(r2_with[i] - r2_without[i]) / abs(r2_without[i]) * 100 for i in range(len(models))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    bar_width = 0.35
    index = np.arange(len(models))
    
    bars1 = axes[0, 0].bar(index, mse_without, bar_width, label='Without Sentiment', color='skyblue')
    bars2 = axes[0, 0].bar(index + bar_width, mse_with, bar_width, label='With Sentiment', color='salmon')
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('MSE (Lower is better)')
    axes[0, 0].set_title('MSE Comparison')
    axes[0, 0].set_xticks(index + bar_width / 2)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
                
    for bar in bars2:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)

    bars3 = axes[0, 1].bar(index, r2_without, bar_width, label='Without Sentiment', color='skyblue')
    bars4 = axes[0, 1].bar(index + bar_width, r2_with, bar_width, label='With Sentiment', color='salmon')
    
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('R² (Higher is better)')
    axes[0, 1].set_title('R² Comparison')
    axes[0, 1].set_xticks(index + bar_width / 2)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars3:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
                
    for bar in bars4:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
    colors = ['green' if x < 0 else 'red' for x in mse_change]
    bars5 = axes[1, 0].bar(models, mse_change, color=colors)
    
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('MSE % Change\n(Negative is better)')
    axes[1, 0].set_title('Impact of Sentiment Data on MSE')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars5:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                f'{height:.2f}%', ha='center', va=va, fontsize=9)
    
    colors = ['green' if x > 0 else 'red' for x in r2_change]
    bars6 = axes[1, 1].bar(models, r2_change, color=colors)
    
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('R² % Change\n(Positive is better)')
    axes[1, 1].set_title('Impact of Sentiment Data on R²')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars6:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                f'{height:.2f}%', ha='center', va=va, fontsize=9)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, len(models)*0.5 + 2))
    
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'MSE\n(Without Sentiment)', 'MSE\n(With Sentiment)', 'MSE\nChange %', 
               'R²\n(Without Sentiment)', 'R²\n(With Sentiment)', 'R²\nChange %']
    
    for i in range(len(models)):
        table_data.append([
            models[i], 
            f"{mse_without[i]:.4f}", 
            f"{mse_with[i]:.4f}", 
            f"{mse_change[i]:.2f}%",
            f"{r2_without[i]:.4f}", 
            f"{r2_with[i]:.4f}", 
            f"{r2_change[i]:.2f}%"
        ])
    
    avg_mse_without = np.mean(mse_without)
    avg_mse_with = np.mean(mse_with)
    avg_mse_change = np.mean(mse_change)
    avg_r2_without = np.mean(r2_without)
    avg_r2_with = np.mean(r2_with)
    avg_r2_change = np.mean(r2_change)
    
    table_data.append([
        'Average', 
        f"{avg_mse_without:.4f}", 
        f"{avg_mse_with:.4f}", 
        f"{avg_mse_change:.2f}%",
        f"{avg_r2_without:.4f}", 
        f"{avg_r2_with:.4f}", 
        f"{avg_r2_change:.2f}%"
    ])
    
    table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == len(models) + 1:
            cell.set_text_props(fontweight='bold')
        
        if col == 3:
            if "%" in cell.get_text().get_text():
                value = float(cell.get_text().get_text().replace('%', ''))
                if value < 0:
                    cell.set_facecolor('#d5f5d5')
                elif value > 0:
                    cell.set_facecolor('#f5d5d5')
        
        if col == 6:  # R²变化列
            if "%" in cell.get_text().get_text():
                value = float(cell.get_text().get_text().replace('%', ''))
                if value > 0:
                    cell.set_facecolor('#d5f5d5')
                elif value < 0:
                    cell.set_facecolor('#f5d5d5')
    
    plt.title("Detailed Performance Comparison: With vs. Without Sentiment Data", pad=20)
    plt.savefig("Performance_Comparison_Table.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- SUMMARY OF SENTIMENT IMPACT ---")
    print(f"Average MSE change: {avg_mse_change:.2f}% {'(Improved)' if avg_mse_change < 0 else '(Worsened)'}")
    print(f"Average R² change: {avg_r2_change:.2f}% {'(Improved)' if avg_r2_change > 0 else '(Worsened)'}")
    
    best_mse_idx = np.argmin(mse_change)
    worst_mse_idx = np.argmax(mse_change)
    best_r2_idx = np.argmax(r2_change)
    worst_r2_idx = np.argmin(r2_change)
    
    print(f"\nBest MSE improvement: {models[best_mse_idx]} ({mse_change[best_mse_idx]:.2f}%)")
    print(f"Worst MSE change: {models[worst_mse_idx]} ({mse_change[worst_mse_idx]:.2f}%)")
    print(f"Best R² improvement: {models[best_r2_idx]} ({r2_change[best_r2_idx]:.2f}%)")
    print(f"Worst R² change: {models[worst_r2_idx]} ({r2_change[worst_r2_idx]:.2f}%)")

    return {
        'avg_mse_change': avg_mse_change,
        'avg_r2_change': avg_r2_change,
        'best_model_mse': models[best_mse_idx],
        'best_model_r2': models[best_r2_idx]
    }


def main():
    global SEQ_LEN, BATCH_SIZE
    # Load datasets - use the same path structure as in your original code
    tesla_path = 'data/tesla_enriched_data.csv'
    final_path = 'data/final_dataset.csv'  # Dataset with sentiment scores
    
    df_tesla = pd.read_csv(tesla_path)
    df_final = pd.read_csv(final_path)
    
    # Clean column names
    df_tesla.rename(columns={
        "('Date', '')": 'Published',
        "('Close', 'TSLA')": 'Close',
        "('High', 'TSLA')": 'High',
        "('Low', 'TSLA')": 'Low',
        "('Open', 'TSLA')": 'Open',
        "('Volume', 'TSLA')": 'Volume',
    }, inplace=True)
    
    # Apply same rename to final_data if needed
    # Assuming final_data has the same column names plus sentimental score
    if 'Published' not in df_final.columns and "('Date', '')" in df_final.columns:
        df_final.rename(columns={
            "('Date', '')": 'Published',
            "('Close', 'TSLA')": 'Close',
            "('High', 'TSLA')": 'High',
            "('Low', 'TSLA')": 'Low',
            "('Open', 'TSLA')": 'Open',
            "('Volume', 'TSLA')": 'Volume',
        }, inplace=True)
    
    # Print column names to verify
    print("Tesla data columns:", df_tesla.columns.tolist())
    print("Final data columns:", df_final.columns.tolist())
    
    print(f"Tesla data total rows: {len(df_tesla)}")
    print(f"Final data total rows: {len(df_final)}")
    
    df_tesla_train = df_tesla[df_tesla['Published'] < '2025-01-01']
    df_tesla_test = df_tesla[df_tesla['Published'] >= '2025-01-01']
    
    df_final_train = df_final[df_final['Published'] < '2025-01-01']
    df_final_test = df_final[df_final['Published'] >= '2025-01-01']
    
    print(f"Tesla train set size: {len(df_tesla_train)}, test set size: {len(df_tesla_test)}")
    print(f"Final train set size: {len(df_final_train)}, test set size: {len(df_final_test)}")
    
    # Check for sentiment column
    sentiment_cols = [col for col in df_final.columns if 'Sentiment Score' or 'score_sq' or 'score_cu' or 'score_dev' in col.lower()]
    if not sentiment_cols:
        sentiment_cols = ['Sentiment Score']
    print(f"Using sentiment columns: {sentiment_cols}")
    
    # Feature columns (without sentiment)
    feature_cols_base = df_tesla.columns.drop(['Published', 'Close'])
    
    # Feature columns (with sentiment)
    feature_cols_sentiment = df_final.columns.drop(['Published', 'Close'])
    
    print(f"Base features: {feature_cols_base.tolist()}")
    print(f"Sentiment features: {feature_cols_sentiment.tolist()}")
    
    # Scale features for Tesla data (without sentiment)
    scaler_X_base = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_base = scaler_X_base.fit_transform(df_tesla_train[feature_cols_base])
    y_train = scaler_y.fit_transform(df_tesla_train['Close'].values.reshape(-1, 1)).flatten()
    
    X_test_base = scaler_X_base.transform(df_tesla_test[feature_cols_base])
    y_test = scaler_y.transform(df_tesla_test['Close'].values.reshape(-1, 1)).flatten()
    
    # Scale features for Final data (with sentiment)
    scaler_X_sentiment = MinMaxScaler()
    
    X_train_sentiment = scaler_X_sentiment.fit_transform(df_final_train[feature_cols_sentiment])
    X_test_sentiment = scaler_X_sentiment.transform(df_final_test[feature_cols_sentiment])
    
    if len(X_test_base) <= SEQ_LEN or len(X_test_sentiment) <= SEQ_LEN:
        old_seq_len = SEQ_LEN
        SEQ_LEN = max(1, min(len(X_test_base) // 2, len(X_test_sentiment) // 2))
        print(f"Warning: Test set too small for sequence length {old_seq_len}. Adjusted to {SEQ_LEN}.")
    
    # Create datasets
    print("\nCreating datasets with sequence length:", SEQ_LEN)
    train_dataset_base = SequenceDataset(X_train_base, y_train, SEQ_LEN)
    test_dataset_base = SequenceDataset(X_test_base, y_test, SEQ_LEN)
    
    train_dataset_sentiment = SequenceDataset(X_train_sentiment, y_train, SEQ_LEN)
    test_dataset_sentiment = SequenceDataset(X_test_sentiment, y_test, SEQ_LEN)
    
    # Create data loaders
    train_loader_base = DataLoader(train_dataset_base, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_sentiment = DataLoader(train_dataset_sentiment, batch_size=BATCH_SIZE, shuffle=True)
    
    X_test_tensor_base = torch.stack([x[0] for x in test_dataset_base])
    X_test_tensor_sentiment = torch.stack([x[0] for x in test_dataset_sentiment])
    y_test_tensor = torch.stack([x[1] for x in test_dataset_base])
    
    # Get input sizes
    input_size_base = X_train_base.shape[1]
    input_size_sentiment = X_train_sentiment.shape[1]
    
    print(f"Input size without sentiment: {input_size_base}")
    print(f"Input size with sentiment: {input_size_sentiment}")
    
    # Keep the original RNN and LSTM configurations
    rnn_configs = [
        {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "lr": 0.005, "opt": 'adam'},
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.1, "lr": 0.001, "opt": 'adam'},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "lr": 0.0005, "opt": 'sgd'},
        {"hidden_size": 256, "num_layers": 3, "dropout": 0.3, "lr": 0.0001, "opt": 'adam'},
        {"hidden_size": 64, "num_layers": 1, "dropout": 0.0, "lr": 0.01, "opt": 'sgd'},
    ]
    
    lstm_configs = rnn_configs.copy()
    
    # Results container
    results = {
        'without_sentiment': {},
        'with_sentiment': {}
    }
    
    # Train and evaluate original RNN models (without sentiment)
    print("\n--- ORIGINAL RNN MODELS (Without Sentiment) ---")
    for i, config in enumerate(rnn_configs):
        print(f"\nTraining RNN Model {i+1} with config: {config}")
        model_params = {k: v for k, v in config.items() if k in ['hidden_size', 'num_layers', 'dropout']}
        rnn_model = CustomRNN(input_size=input_size_base, **model_params)
        trained_model, losses = train_model(rnn_model, train_loader_base, lr=config["lr"], optimizer_type=config["opt"])
        preds, true, mse, r2 = evaluate_model(trained_model, X_test_tensor_base, y_test_tensor)
        print(f"[RNN {i+1}] MSE: {mse:.4f}, R^2: {r2:.4f}")
        
        # Store results
        results['without_sentiment'][f'RNN_{i+1}'] = {
            'config': config,
            'true': true,
            'predictions': preds,
            'metrics': {'mse': mse, 'r2': r2},
            'losses': losses
        }
        
        # Plot predictions
        plt.figure(figsize=(10, 4))
        plt.plot(true, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.title(f"RNN Model {i+1} (Without Sentiment)")
        plt.xlabel("# of Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig(f"RNN_Model_{i+1}_without_sentiment.png", dpi=300)
        plt.close()
    
    # Train and evaluate original LSTM models (without sentiment)
    print("\n--- ORIGINAL LSTM MODELS (Without Sentiment) ---")
    for i, config in enumerate(lstm_configs):
        print(f"\nTraining LSTM Model {i+1} with config: {config}")
        model_params = {k: v for k, v in config.items() if k in ['hidden_size', 'num_layers', 'dropout']}
        lstm_model = CustomLSTM(input_size=input_size_base, **model_params)
        trained_model, losses = train_model(lstm_model, train_loader_base, lr=config["lr"], optimizer_type=config["opt"])
        preds, true, mse, r2 = evaluate_model(trained_model, X_test_tensor_base, y_test_tensor)
        print(f"[LSTM {i+1}] MSE: {mse:.4f}, R^2: {r2:.4f}")
        
        # Store results
        results['without_sentiment'][f'LSTM_{i+1}'] = {
            'config': config,
            'true': true,
            'predictions': preds,
            'metrics': {'mse': mse, 'r2': r2},
            'losses': losses
        }
        
        # Plot predictions
        plt.figure(figsize=(10, 4))
        plt.plot(true, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.title(f"LSTM Model {i+1} (Without Sentiment)")
        plt.xlabel("# of Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig(f"LSTM_Model_{i+1}_without_sentiment.png", dpi=300)
        plt.close()
    
    # Find the best performing models without sentiment
    best_rnn_model = min(
        [(f'RNN_{i+1}', results['without_sentiment'][f'RNN_{i+1}']['metrics']['mse']) for i in range(len(rnn_configs))],
        key=lambda x: x[1]
    )[0]
    
    best_lstm_model = min(
        [(f'LSTM_{i+1}', results['without_sentiment'][f'LSTM_{i+1}']['metrics']['mse']) for i in range(len(lstm_configs))],
        key=lambda x: x[1]
    )[0]
    
    print(f"\nBest RNN model without sentiment: {best_rnn_model}")
    print(f"Best LSTM model without sentiment: {best_lstm_model}")
    
    # Now train models with sentiment data using the same configurations
    print("\n--- RNN MODELS WITH SENTIMENT ---")
    for i, config in enumerate(rnn_configs):
        print(f"\nTraining RNN Model {i+1} with config: {config} (with sentiment)")
        model_params = {k: v for k, v in config.items() if k in ['hidden_size', 'num_layers', 'dropout']}
        rnn_model = CustomRNN(input_size=input_size_sentiment, **model_params)
        trained_model, losses = train_model(rnn_model, train_loader_sentiment, lr=config["lr"], optimizer_type=config["opt"])
        preds, true, mse, r2 = evaluate_model(trained_model, X_test_tensor_sentiment, y_test_tensor)
        print(f"[RNN {i+1} with sentiment] MSE: {mse:.4f}, R^2: {r2:.4f}")
        
        # Store results
        results['with_sentiment'][f'RNN_{i+1}'] = {
            'config': config,
            'true': true,
            'predictions': preds,
            'metrics': {'mse': mse, 'r2': r2},
            'losses': losses
        }
        
        # Plot predictions
        plt.figure(figsize=(10, 4))
        plt.plot(true, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.title(f"RNN Model {i+1} (With Sentiment)")
        plt.xlabel("# of Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig(f"RNN_Model_{i+1}_with_sentiment.png", dpi=300)
        plt.close()
        
        # Create comparison plot
        plt.figure(figsize=(12, 5))
        plt.plot(results['without_sentiment'][f'RNN_{i+1}']['true'], label="Actual", linewidth=2)
        plt.plot(results['without_sentiment'][f'RNN_{i+1}']['predictions'], label=f"Without Sentiment (MSE: {results['without_sentiment'][f'RNN_{i+1}']['metrics']['mse']:.4f})", linestyle='--')
        plt.plot(results['with_sentiment'][f'RNN_{i+1}']['predictions'], label=f"With Sentiment (MSE: {results['with_sentiment'][f'RNN_{i+1}']['metrics']['mse']:.4f})", linestyle='-.')
        plt.title(f"RNN Model {i+1} - Sentiment Impact Comparison")
        plt.xlabel("# of Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig(f"RNN_Model_{i+1}_comparison.png", dpi=300)
        plt.close()
    
    print("\n--- LSTM MODELS WITH SENTIMENT ---")
    for i, config in enumerate(lstm_configs):
        print(f"\nTraining LSTM Model {i+1} with config: {config} (with sentiment)")
        model_params = {k: v for k, v in config.items() if k in ['hidden_size', 'num_layers', 'dropout']}
        lstm_model = CustomLSTM(input_size=input_size_sentiment, **model_params)
        trained_model, losses = train_model(lstm_model, train_loader_sentiment, lr=config["lr"], optimizer_type=config["opt"])
        preds, true, mse, r2 = evaluate_model(trained_model, X_test_tensor_sentiment, y_test_tensor)
        print(f"[LSTM {i+1} with sentiment] MSE: {mse:.4f}, R^2: {r2:.4f}")
        
        # Store results
        results['with_sentiment'][f'LSTM_{i+1}'] = {
            'config': config,
            'true': true,
            'predictions': preds,
            'metrics': {'mse': mse, 'r2': r2},
            'losses': losses
        }
        
        # Plot predictions
        plt.figure(figsize=(10, 4))
        plt.plot(true, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.title(f"LSTM Model {i+1} (With Sentiment)")
        plt.xlabel("# of Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig(f"LSTM_Model_{i+1}_with_sentiment.png", dpi=300)
        plt.close()
        
        # Create comparison plot
        plt.figure(figsize=(12, 5))
        plt.plot(results['without_sentiment'][f'LSTM_{i+1}']['true'], label="Actual", linewidth=2)
        plt.plot(results['without_sentiment'][f'LSTM_{i+1}']['predictions'], label=f"Without Sentiment (MSE: {results['without_sentiment'][f'LSTM_{i+1}']['metrics']['mse']:.4f})", linestyle='--')
        plt.plot(results['with_sentiment'][f'LSTM_{i+1}']['predictions'], label=f"With Sentiment (MSE: {results['with_sentiment'][f'LSTM_{i+1}']['metrics']['mse']:.4f})", linestyle='-.')
        plt.title(f"LSTM Model {i+1} - Sentiment Impact Comparison")
        plt.xlabel("# of Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig(f"LSTM_Model_{i+1}_comparison.png", dpi=300)
        plt.close()
    
    # Find the best performing models with sentiment
    best_rnn_model_sentiment = min(
        [(f'RNN_{i+1}', results['with_sentiment'][f'RNN_{i+1}']['metrics']['mse']) for i in range(len(rnn_configs))],
        key=lambda x: x[1]
    )[0]
    
    best_lstm_model_sentiment = min(
        [(f'LSTM_{i+1}', results['with_sentiment'][f'LSTM_{i+1}']['metrics']['mse']) for i in range(len(lstm_configs))],
        key=lambda x: x[1]
    )[0]
    
    print(f"\nBest RNN model with sentiment: {best_rnn_model_sentiment}")
    print(f"Best LSTM model with sentiment: {best_lstm_model_sentiment}")
    
    # Collect comparison results
    comparison_results = {
        'RNN': {
            'true': results['without_sentiment'][best_rnn_model]['true'],
            'predictions': {
                'without sentiment': results['without_sentiment'][best_rnn_model]['predictions'],
                'with sentiment': results['with_sentiment'][best_rnn_model_sentiment]['predictions']
            },
            'metrics': {
                'without sentiment': results['without_sentiment'][best_rnn_model]['metrics'],
                'with sentiment': results['with_sentiment'][best_rnn_model_sentiment]['metrics']
            },
            'losses': {
                'without sentiment': results['without_sentiment'][best_rnn_model]['losses'],
                'with sentiment': results['with_sentiment'][best_rnn_model_sentiment]['losses']
            }
        },
        'LSTM': {
            'true': results['without_sentiment'][best_lstm_model]['true'],
            'predictions': {
                'without sentiment': results['without_sentiment'][best_lstm_model]['predictions'],
                'with sentiment': results['with_sentiment'][best_lstm_model_sentiment]['predictions']
            },
            'metrics': {
                'without sentiment': results['without_sentiment'][best_lstm_model]['metrics'],
                'with sentiment': results['with_sentiment'][best_lstm_model_sentiment]['metrics']
            },
            'losses': {
                'without sentiment': results['without_sentiment'][best_lstm_model]['losses'],
                'with sentiment': results['with_sentiment'][best_lstm_model_sentiment]['losses']
            }
        }
    }
    
    # Visualize predictions
    plot_predictions(results, title="Sentiment Impact on Stock Prediction")
    
    # Plot comparative metrics
    plot_comparison_metrics(results)
    
    # Plot training losses
    plt.figure(figsize=(12, 6))
    
    # RNN losses
    plt.subplot(1, 2, 1)
    plt.plot(results['RNN']['losses']['without sentiment'], label='Without Sentiment', color='blue')
    plt.plot(results['RNN']['losses']['with sentiment'], label='With Sentiment', color='red')
    plt.title('RNN Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # LSTM losses
    plt.subplot(1, 2, 2)
    plt.plot(results['LSTM']['losses']['without sentiment'], label='Without Sentiment', color='blue')
    plt.plot(results['LSTM']['losses']['with sentiment'], label='With Sentiment', color='red')
    plt.title('LSTM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Training_Losses.png", dpi=300)
    plt.show()
    
    # Calculate percentage improvements
    rnn_mse_improvement = ((rnn_base_mse - rnn_sentiment_mse) / rnn_base_mse) * 100
    rnn_r2_improvement = ((rnn_sentiment_r2 - rnn_base_r2) / abs(rnn_base_r2)) * 100
    
    lstm_mse_improvement = ((lstm_base_mse - lstm_sentiment_mse) / lstm_base_mse) * 100
    lstm_r2_improvement = ((lstm_sentiment_r2 - lstm_base_r2) / abs(lstm_base_r2)) * 100
    
    # Print summary
    print("\n--- SUMMARY OF SENTIMENT IMPACT ---")
    print(f"RNN Improvement with Sentiment: MSE reduced by {rnn_mse_improvement:.2f}%, R² improved by {rnn_r2_improvement:.2f}%")
    print(f"LSTM Improvement with Sentiment: MSE reduced by {lstm_mse_improvement:.2f}%, R² improved by {lstm_r2_improvement:.2f}%")
    
    # Create a heatmap to visualize feature importance
    if len(sentiment_cols) > 0:
        # Display correlation with sentiment scores
        corr_matrix = df_final.corr()
        sentiment_corr = corr_matrix.loc[sentiment_cols, :].drop(sentiment_cols, axis=1)
        
        plt.figure(figsize=(12, len(sentiment_cols) * 2))
        plt.imshow(sentiment_corr, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation')
        
        # Add the values to the heatmap
        for i in range(sentiment_corr.shape[0]):
            for j in range(sentiment_corr.shape[1]):
                value = sentiment_corr.iloc[i, j]
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                         color='white' if abs(value) > 0.5 else 'black')
        
        plt.xticks(range(len(sentiment_corr.columns)), sentiment_corr.columns, rotation=90)
        plt.yticks(range(len(sentiment_corr.index)), sentiment_corr.index)
        plt.title('Correlation of Sentiment Scores with Stock Features')
        plt.tight_layout()
        plt.savefig("Sentiment_Correlation.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    # Run main function
    main()