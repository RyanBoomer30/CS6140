import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def prepare_data(df):
    features = ['Open', 'Sentiment Score']
    targets = ['High', 'Low', 'Close', 'Volume']
    
    X = df[features].values
    y = df[targets].values
    
    return X, y, df['Published'].values

def plot_correlation_matrix(df, columns=None):
    if columns is None:
        columns = df.columns
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
    
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    plt.show()

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DNN(nn.Module):
    def __init__(self, input_size, output_size=4, hidden_sizes=[64, 32], dropout_rate=0.2):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class CustomRNN(nn.Module):
    def __init__(self, input_size, output_size=4, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class CustomLSTM(nn.Module):
    def __init__(self, input_size, output_size=4, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    

def train_model(model, train_loader, val_loader=None, epochs=50, lr=0.001):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
    
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test, is_nn=True, target_names=None, model_name="Model"):
    if is_nn:
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            preds = model(X_test_tensor).cpu().numpy()
    else:
        preds = np.array([model[i].predict(X_test) for i in range(len(model))]).T
    
    target_metrics = []
    
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:, i], preds[:, i])
        r2 = r2_score(y_test[:, i], preds[:, i])
        
        target_name = target_names[i] if target_names else f"Target {i+1}"
        print(f"{model_name} - {target_name}: MSE={mse:.6f}, R²={r2:.6f}")
        
        target_metrics.append({"MSE": mse, "R2": r2})
    
    avg_mse = np.mean([m["MSE"] for m in target_metrics])
    avg_r2 = np.mean([m["R2"] for m in target_metrics])
    
    print(f"{model_name} - Average: MSE={avg_mse:.6f}, R²={avg_r2:.6f}")
    
    return preds, target_metrics, avg_mse, avg_r2

def plot_predictions(preds, y_true, dates=None, target_names=None, model_name="Model"):
    n_targets = y_true.shape[1]
    target_names = target_names or [f"Target {i+1}" for i in range(n_targets)]
    
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))
    
    for i in range(n_targets):
        ax = axes[i] if n_targets > 1 else axes
        
        if dates is not None:
            ax.plot(dates, y_true[:, i], label="Actual", linewidth=2)
            ax.plot(dates, preds[:, i], label="Predicted", linewidth=2, linestyle='--')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.plot(y_true[:, i], label="Actual", linewidth=2)
            ax.plot(preds[:, i], label="Predicted", linewidth=2, linestyle='--')
        
        ax.set_title(f"{model_name} - {target_names[i]} Prediction", fontsize=15)
        ax.set_xlabel('Date' if dates else 'Time Steps', fontsize=12)
        ax.set_ylabel(target_names[i], fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_results, metric='MSE', target_idx=None, title=None):
    plt.figure(figsize=(12, 6))
    
    if target_idx is not None:
        models = list(model_results.keys())
        values = [model_results[model]["target_metrics"][target_idx][metric] for model in models]
        
        plt.bar(models, values)
        plt.title(title or f"{metric} Comparison - {model_results[models[0]]['target_names'][target_idx]}", fontsize=15)
    else:
        models = list(model_results.keys())
        values = [model_results[model][f"avg_{metric.lower()}"] for model in models]
        
        plt.bar(models, values)
        plt.title(title or f"Average {metric} Comparison", fontsize=15)
    
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        plt.text(i, v * 1.05, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()

def cross_validate(X, y, model_class, model_params=None, n_splits=5, epochs=50, lr=0.001, is_nn=True, target_names=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_targets = y.shape[1]
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_splits}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        if is_nn:
            train_dataset = StockDataset(X_train_fold, y_train_fold)
            val_dataset = StockDataset(X_val_fold, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            model = model_class(**(model_params or {}))
            
            trained_model, _, _ = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
            
            _, target_metrics, avg_mse, avg_r2 = evaluate_model(
                trained_model, X_val_fold, y_val_fold, 
                is_nn=True, target_names=target_names,
                model_name=f"Fold {fold+1}"
            )
        else:
            lr_models = []
            target_metrics = []
            
            for i in range(n_targets):
                lr_model = LinearRegression(**(model_params or {}))
                lr_model.fit(X_train_fold, y_train_fold[:, i])
                lr_models.append(lr_model)
                
                preds = lr_model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold[:, i], preds)
                r2 = r2_score(y_val_fold[:, i], preds)
                
                target_name = target_names[i] if target_names else f"Target {i+1}"
                print(f"Fold {fold+1} - {target_name}: MSE={mse:.6f}, R²={r2:.6f}")
                
                target_metrics.append({"MSE": mse, "R2": r2})
            
            avg_mse = np.mean([m["MSE"] for m in target_metrics])
            avg_r2 = np.mean([m["R2"] for m in target_metrics])
            
            print(f"Fold {fold+1} - Average: MSE={avg_mse:.6f}, R²={avg_r2:.6f}")
        
        fold_metrics.append({"target_metrics": target_metrics, "avg_mse": avg_mse, "avg_r2": avg_r2})
    
    avg_target_mse = [np.mean([fold_metrics[j]["target_metrics"][i]["MSE"] for j in range(n_splits)]) for i in range(n_targets)]
    avg_target_r2 = [np.mean([fold_metrics[j]["target_metrics"][i]["R2"] for j in range(n_splits)]) for i in range(n_targets)]
    
    std_target_mse = [np.std([fold_metrics[j]["target_metrics"][i]["MSE"] for j in range(n_splits)]) for i in range(n_targets)]
    std_target_r2 = [np.std([fold_metrics[j]["target_metrics"][i]["R2"] for j in range(n_splits)]) for i in range(n_targets)]
    
    avg_mse = np.mean([fold_metrics[i]["avg_mse"] for i in range(n_splits)])
    avg_r2 = np.mean([fold_metrics[i]["avg_r2"] for i in range(n_splits)])
    
    std_mse = np.std([fold_metrics[i]["avg_mse"] for i in range(n_splits)])
    std_r2 = np.std([fold_metrics[i]["avg_r2"] for i in range(n_splits)])
    
    print("\nCross-Validation Results:")
    for i in range(n_targets):
        target_name = target_names[i] if target_names else f"Target {i+1}"
        print(f"{target_name}: MSE={avg_target_mse[i]:.6f}±{std_target_mse[i]:.6f}, R²={avg_target_r2[i]:.6f}±{std_target_r2[i]:.6f}")
    
    print(f"Average: MSE={avg_mse:.6f}±{std_mse:.6f}, R²={avg_r2:.6f}±{std_r2:.6f}")
    
    return {
        "avg_target_mse": avg_target_mse,
        "avg_target_r2": avg_target_r2,
        "std_target_mse": std_target_mse,
        "std_target_r2": std_target_r2,
        "avg_mse": avg_mse,
        "avg_r2": avg_r2,
        "std_mse": std_mse,
        "std_r2": std_r2
    }

def main():
    df = load_data('data/final_dataset.csv')
    
    print("\n===== Data Correlation Analysis =====")
    cols_to_analyze = ['Sentiment Score', 'Open', 'High', 'Low', 'Close', 'Volume']
    plot_correlation_matrix(df, cols_to_analyze)
    
    X, y, dates = prepare_data(df)
    print(f"Features shape: {X.shape}, Targets shape: {y.shape}")
    
    target_names = ['High', 'Low', 'Close', 'Volume']
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[train_size:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    train_dataset = StockDataset(X_train_scaled, y_train_scaled)
    test_dataset = StockDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    all_results = {}
    
    print("\n===== Linear Regression =====")
    
    lr_models = []
    for i in range(len(target_names)):
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train_scaled[:, i])
        lr_models.append(lr_model)
    
    lr_preds, lr_target_metrics, lr_avg_mse, lr_avg_r2 = evaluate_model(
        lr_models, X_test_scaled, y_test_scaled, 
        is_nn=False, target_names=target_names,
        model_name="Linear Regression"
    )
    
    all_results["Linear Regression"] = {
        "preds": lr_preds,
        "target_metrics": lr_target_metrics,
        "avg_mse": lr_avg_mse,
        "avg_r2": lr_avg_r2,
        "target_names": target_names
    }
    
    lr_preds_orig = scaler_y.inverse_transform(lr_preds)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)
    
    plot_predictions(lr_preds_orig, y_test_orig, test_dates, target_names, "Linear Regression")
    
    print("\n===== DNN Model =====")
    
    input_size = X_train_scaled.shape[1]
    output_size = y_train_scaled.shape[1]
    
    dnn_model = DNN(input_size=input_size, output_size=output_size)
    dnn_model, _, _ = train_model(dnn_model, train_loader, epochs=50)
    
    dnn_preds, dnn_target_metrics, dnn_avg_mse, dnn_avg_r2 = evaluate_model(
        dnn_model, X_test_scaled, y_test_scaled, 
        is_nn=True, target_names=target_names,
        model_name="DNN"
    )
    
    all_results["DNN"] = {
        "preds": dnn_preds,
        "target_metrics": dnn_target_metrics,
        "avg_mse": dnn_avg_mse,
        "avg_r2": dnn_avg_r2,
        "target_names": target_names
    }
    
    dnn_preds_orig = scaler_y.inverse_transform(dnn_preds)
    plot_predictions(dnn_preds_orig, y_test_orig, test_dates, target_names, "DNN")
    
    print("\n--- DNN Hyperparameter Testing ---")
    
    dnn_deep = DNN(input_size=input_size, output_size=output_size, hidden_sizes=[128, 64, 32])
    dnn_deep, _, _ = train_model(dnn_deep, train_loader, epochs=50)
    
    dnn_deep_preds, dnn_deep_target_metrics, dnn_deep_avg_mse, dnn_deep_avg_r2 = evaluate_model(
        dnn_deep, X_test_scaled, y_test_scaled, 
        is_nn=True, target_names=target_names,
        model_name="DNN (Deep)"
    )
    
    all_results["DNN (Deep)"] = {
        "preds": dnn_deep_preds,
        "target_metrics": dnn_deep_target_metrics,
        "avg_mse": dnn_deep_avg_mse,
        "avg_r2": dnn_deep_avg_r2,
        "target_names": target_names
    }
    
    print("\n===== RNN Model =====")
    
    rnn_model = CustomRNN(input_size=input_size, output_size=output_size)
    rnn_model, _, _ = train_model(rnn_model, train_loader, epochs=50)
    
    rnn_preds, rnn_target_metrics, rnn_avg_mse, rnn_avg_r2 = evaluate_model(
        rnn_model, X_test_scaled, y_test_scaled, 
        is_nn=True, target_names=target_names,
        model_name="RNN"
    )
    
    all_results["RNN"] = {
        "preds": rnn_preds,
        "target_metrics": rnn_target_metrics,
        "avg_mse": rnn_avg_mse,
        "avg_r2": rnn_avg_r2,
        "target_names": target_names
    }
    
    rnn_preds_orig = scaler_y.inverse_transform(rnn_preds)
    plot_predictions(rnn_preds_orig, y_test_orig, test_dates, target_names, "RNN")
    
    print("\n--- RNN Hyperparameter Testing ---")
    
    rnn_large = CustomRNN(input_size=input_size, output_size=output_size, hidden_size=128)
    rnn_large, _, _ = train_model(rnn_large, train_loader, epochs=50)
    
    rnn_large_preds, rnn_large_target_metrics, rnn_large_avg_mse, rnn_large_avg_r2 = evaluate_model(
        rnn_large, X_test_scaled, y_test_scaled, 
        is_nn=True, target_names=target_names,
        model_name="RNN (Large Hidden)"
    )
    
    all_results["RNN (Large Hidden)"] = {
        "preds": rnn_large_preds,
        "target_metrics": rnn_large_target_metrics,
        "avg_mse": rnn_large_avg_mse,
        "avg_r2": rnn_large_avg_r2,
        "target_names": target_names
    }
    
    print("\n===== LSTM Model =====")
    
    lstm_model = CustomLSTM(input_size=input_size, output_size=output_size)
    lstm_model, _, _ = train_model(lstm_model, train_loader, epochs=50)
    
    lstm_preds, lstm_target_metrics, lstm_avg_mse, lstm_avg_r2 = evaluate_model(
        lstm_model, X_test_scaled, y_test_scaled, 
        is_nn=True, target_names=target_names,
        model_name="LSTM"
    )
    
    all_results["LSTM"] = {
        "preds": lstm_preds,
        "target_metrics": lstm_target_metrics,
        "avg_mse": lstm_avg_mse,
        "avg_r2": lstm_avg_r2,
        "target_names": target_names
    }
    
    lstm_preds_orig = scaler_y.inverse_transform(lstm_preds)
    plot_predictions(lstm_preds_orig, y_test_orig, test_dates, target_names, "LSTM")
    
    print("\n--- LSTM Hyperparameter Testing ---")
    
    lstm_multi = CustomLSTM(input_size=input_size, output_size=output_size, num_layers=2, dropout=0.2)
    lstm_multi, _, _ = train_model(lstm_multi, train_loader, epochs=50)
    
    lstm_multi_preds, lstm_multi_target_metrics, lstm_multi_avg_mse, lstm_multi_avg_r2 = evaluate_model(
        lstm_multi, X_test_scaled, y_test_scaled, 
        is_nn=True, target_names=target_names,
        model_name="LSTM (Multilayer)"
    )
    
    all_results["LSTM (Multilayer)"] = {
        "preds": lstm_multi_preds,
        "target_metrics": lstm_multi_target_metrics,
        "avg_mse": lstm_multi_avg_mse,
        "avg_r2": lstm_multi_avg_r2,
        "target_names": target_names
    }
    
    print("\n===== Model Comparison =====")
    
    for model_name, results in all_results.items():
        print(f"{model_name}: Avg MSE={results['avg_mse']:.6f}, Avg R²={results['avg_r2']:.6f}")
    
    plot_model_comparison(all_results, 'MSE', None, "Average MSE Comparison (Lower is Better)")
    
    plot_model_comparison(all_results, 'R2', None, "Average R² Comparison (Higher is Better)")
    
    for i, target_name in enumerate(target_names):
        plot_model_comparison(all_results, 'MSE', i, f"{target_name} - MSE Comparison (Lower is Better)")
        plot_model_comparison(all_results, 'R2', i, f"{target_name} - R² Comparison (Higher is Better)")
    
    print("\n===== Cross-Validation =====")
    
    print("\n--- Linear Regression Cross-Validation ---")
    lr_cv_results = cross_validate(
        X_train_scaled, y_train_scaled, LinearRegression, 
        is_nn=False, target_names=target_names
    )
    
    print("\n--- DNN Cross-Validation ---")
    dnn_cv_results = cross_validate(
        X_train_scaled, y_train_scaled, DNN, 
        {"input_size": input_size, "output_size": output_size},
        is_nn=True, target_names=target_names
    )
    
    print("\n--- RNN Cross-Validation ---")
    rnn_cv_results = cross_validate(
        X_train_scaled, y_train_scaled, CustomRNN, 
        {"input_size": input_size, "output_size": output_size},
        is_nn=True, target_names=target_names
    )
    
    print("\n--- LSTM Cross-Validation ---")
    lstm_cv_results = cross_validate(
        X_train_scaled, y_train_scaled, CustomLSTM, 
        {"input_size": input_size, "output_size": output_size},
        is_nn=True, target_names=target_names
    )
    
    print("\n===== Cross-Validation Comparison =====")
    cv_models = ["Linear Regression", "DNN", "RNN", "LSTM"]
    cv_results = [lr_cv_results, dnn_cv_results, rnn_cv_results, lstm_cv_results]
    
    plt.figure(figsize=(10, 6))
    avg_mse_values = [res["avg_mse"] for res in cv_results]
    std_mse_values = [res["std_mse"] for res in cv_results]
    
    plt.bar(cv_models, avg_mse_values, yerr=std_mse_values, capsize=5)
    plt.title("Cross-Validation: Average MSE Comparison (Lower is Better)", fontsize=15)
    plt.ylabel("MSE", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_mse_values):
        plt.text(i, v + 0.01*max(avg_mse_values), f"{v:.4f}±{std_mse_values[i]:.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    avg_r2_values = [res["avg_r2"] for res in cv_results]
    std_r2_values = [res["std_r2"] for res in cv_results]
    
    plt.bar(cv_models, avg_r2_values, yerr=std_r2_values, capsize=5)
    plt.title("Cross-Validation: Average R² Comparison (Higher is Better)", fontsize=15)
    plt.ylabel("R²", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_r2_values):
        plt.text(i, v + 0.01, f"{v:.4f}±{std_r2_values[i]:.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print("\n===== Sentiment Score Impact Analysis =====")
    
    sentiment_coef = [model.coef_[1] for model in lr_models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(target_names, sentiment_coef)
    plt.title("Sentiment Score Coefficient in Linear Regression", fontsize=15)
    plt.ylabel("Coefficient Value", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    for i, v in enumerate(sentiment_coef):
        plt.text(i, v + 0.01 if v >= 0 else v - 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 10))
    
    for i, target_name in enumerate(target_names):
        plt.subplot(2, 2, i+1)
        plt.scatter(X_test_scaled[:, 1], y_test_scaled[:, i], label="Actual", alpha=0.7)
        plt.scatter(X_test_scaled[:, 1], lstm_preds[:, i], label="LSTM Predicted", alpha=0.7)
        plt.title(f"Sentiment Score vs {target_name}", fontsize=14)
        plt.xlabel("Sentiment Score (Normalized)", fontsize=12)
        plt.ylabel(f"{target_name} (Normalized)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()