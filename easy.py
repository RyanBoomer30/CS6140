import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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


def plot_comparison(true, preds_open, preds_combined, mse_open, mse_combined, r2_open, r2_combined):
    plt.figure(figsize=(12, 6))
    plt.plot(true, 'k-', label='Actual Close', linewidth=2)
    plt.plot(preds_open, 'b--', label=f'Open Only (MSE: {mse_open:.6f}, R²: {r2_open:.6f})', linewidth=1.5)
    plt.plot(preds_combined, 'r-.', label=f'Open + Sentiment (MSE: {mse_combined:.6f}, R²: {r2_combined:.6f})', linewidth=1.5)
    plt.title('Prediction Comparison: Open vs Open+Sentiment')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Close Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Open_vs_OpenSentiment_Comparison.png", dpi=300)
    plt.show()


def main():
    global SEQ_LEN, BATCH_SIZE
    # Load datasets
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
    
    # Split into train and test sets
    df_tesla_train = df_tesla[df_tesla['Published'] < '2025-01-01']
    df_tesla_test = df_tesla[df_tesla['Published'] >= '2025-01-01']
    
    df_final_train = df_final[df_final['Published'] < '2025-01-01']
    df_final_test = df_final[df_final['Published'] >= '2025-01-01']
    
    print(f"Tesla train set size: {len(df_tesla_train)}, test set size: {len(df_tesla_test)}")
    print(f"Final train set size: {len(df_final_train)}, test set size: {len(df_final_test)}")
    
    # Check for sentiment column
    sentiment_cols = [col for col in df_final.columns if 'senti' in col.lower()]
    if not sentiment_cols:
        sentiment_cols = ['Sentiment Score']
    print(f"Using sentiment columns: {sentiment_cols}")
    
    # Simplified feature sets
    # 1. Only Open
    feature_cols_open = ['Open']
    
    # 2. Open + Sentiment
    feature_cols_combined = ['Open'] + sentiment_cols
    
    print(f"Open only features: {feature_cols_open}")
    print(f"Combined features: {feature_cols_combined}")
    
    # Scale features and target
    scaler_X_open = MinMaxScaler()
    scaler_X_combined = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Prepare Open only data
    X_train_open = scaler_X_open.fit_transform(df_tesla_train[feature_cols_open])
    y_train = scaler_y.fit_transform(df_tesla_train['Close'].values.reshape(-1, 1)).flatten()
    
    X_test_open = scaler_X_open.transform(df_tesla_test[feature_cols_open])
    y_test = scaler_y.transform(df_tesla_test['Close'].values.reshape(-1, 1)).flatten()
    
    # Prepare Open + Sentiment data
    X_train_combined = scaler_X_combined.fit_transform(df_final_train[feature_cols_combined])
    X_test_combined = scaler_X_combined.transform(df_final_test[feature_cols_combined])
    
    # Adjust sequence length if needed
    if min(len(X_test_open), len(X_test_combined)) <= SEQ_LEN:
        old_seq_len = SEQ_LEN
        SEQ_LEN = max(1, min(len(X_test_open) // 2, len(X_test_combined) // 2))
        print(f"Warning: Test set too small. Adjusted sequence length from {old_seq_len} to {SEQ_LEN}.")
    
    # Create datasets
    print("\nCreating datasets with sequence length:", SEQ_LEN)
    train_dataset_open = SequenceDataset(X_train_open, y_train, SEQ_LEN)
    test_dataset_open = SequenceDataset(X_test_open, y_test, SEQ_LEN)
    
    train_dataset_combined = SequenceDataset(X_train_combined, y_train, SEQ_LEN)
    test_dataset_combined = SequenceDataset(X_test_combined, y_test, SEQ_LEN)
    
    # Create data loaders
    train_loader_open = DataLoader(train_dataset_open, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_combined = DataLoader(train_dataset_combined, batch_size=BATCH_SIZE, shuffle=True)
    
    X_test_tensor_open = torch.stack([x[0] for x in test_dataset_open])
    X_test_tensor_combined = torch.stack([x[0] for x in test_dataset_combined])
    y_test_tensor = torch.stack([x[1] for x in test_dataset_open])
    
    # Get input sizes
    input_size_open = X_train_open.shape[1]
    input_size_combined = X_train_combined.shape[1]
    
    print(f"Input size for Open only: {input_size_open}")
    print(f"Input size for Open + Sentiment: {input_size_combined}")
    
    # Define model configurations
    lstm_config = {
        "hidden_size": 64, 
        "num_layers": 2, 
        "dropout": 0.1, 
        "lr": 0.001, 
        "opt": 'adam'
    }
    
    # Train model with Open only
    print("\n--- Training LSTM with Open only ---")
    start_time = time.time()
    model_params = {k: v for k, v in lstm_config.items() if k in ['hidden_size', 'num_layers', 'dropout']}
    
    lstm_open = CustomLSTM(input_size=input_size_open, **model_params)
    lstm_open_trained, losses_open = train_model(
        lstm_open, 
        train_loader_open, 
        lr=lstm_config["lr"], 
        optimizer_type=lstm_config["opt"]
    )
    
    preds_open, true_open, mse_open, r2_open = evaluate_model(lstm_open_trained, X_test_tensor_open, y_test_tensor)
    print(f"Open only - MSE: {mse_open:.6f}, R²: {r2_open:.6f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Train model with Open + Sentiment
    print("\n--- Training LSTM with Open + Sentiment ---")
    start_time = time.time()
    
    lstm_combined = CustomLSTM(input_size=input_size_combined, **model_params)
    lstm_combined_trained, losses_combined = train_model(
        lstm_combined, 
        train_loader_combined, 
        lr=lstm_config["lr"], 
        optimizer_type=lstm_config["opt"]
    )
    
    preds_combined, true_combined, mse_combined, r2_combined = evaluate_model(
        lstm_combined_trained, 
        X_test_tensor_combined, 
        y_test_tensor
    )
    print(f"Open + Sentiment - MSE: {mse_combined:.6f}, R²: {r2_combined:.6f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Calculate improvement
    mse_improvement = ((mse_open - mse_combined) / mse_open) * 100
    r2_improvement = ((r2_combined - r2_open) / abs(r2_open)) * 100 if r2_open != 0 else float('inf')
    
    print("\n--- SUMMARY OF SENTIMENT IMPACT ---")
    print(f"MSE change: {mse_improvement:.2f}% {'(Improved)' if mse_improvement > 0 else '(Worsened)'}")
    print(f"R² change: {r2_improvement:.2f}% {'(Improved)' if r2_improvement > 0 else '(Worsened)'}")
    
    # Visualize predictions comparison
    plot_comparison(
        true_open, 
        preds_open, 
        preds_combined, 
        mse_open, 
        mse_combined, 
        r2_open, 
        r2_combined
    )
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses_open, 'b-', label='Open Only')
    plt.plot(losses_combined, 'r-', label='Open + Sentiment')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Training_Losses_Comparison.png", dpi=300)
    plt.show()
    
    # Create correlation plot if sentiment data exists
    if len(sentiment_cols) > 0:
        corr_matrix = df_final[['Open', 'Close'] + sentiment_cols].corr()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation')
        
        # Add correlation values
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                value = corr_matrix.iloc[i, j]
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                         color='white' if abs(value) > 0.5 else 'black')
        
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Correlation between Open, Close and Sentiment')
        plt.tight_layout()
        plt.savefig("Open_Close_Sentiment_Correlation.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    # Run main function
    main()