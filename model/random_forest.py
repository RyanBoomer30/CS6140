import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load the merged dataset and parse dates
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
df = pd.read_csv(DATA_DIR / 'final_dataset.csv', parse_dates=['Published'])

# 2. Split into train/test by date
cutoff = pd.Timestamp('2025-01-01')
df = df.sort_values('Published')
df_train = df.loc[df.Published < cutoff]
df_test  = df.loc[df.Published >= cutoff]

print(df_train.size)
print(df_test.size)

# 3. Define feature columns and target
features = [
    'High', 'Low', 'Open', 'Volume',
    'score_sq', 'score_cu', 'score_dev',
    'inverted_score', 'skewed_score', 'inverted_skewed_score'
]

X_train = df_train[features]
y_train = df_train['Close']
X_test  = df_test[features]
y_test  = df_test['Close']

# 4. Define five different RF configurations
rf_configs = [
    {
        'name': 'RF_Default',
        'n_estimators': 100,
        'random_state': 42
    },
    {
        'name': 'RF_Deeper',
        'n_estimators': 200,
        'max_depth': 20,
        'random_state': 42
    },
    {
        'name': 'RF_Shallow',
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_split': 10,
        'random_state': 42
    },
    {
        'name': 'RF_ManyTrees',
        'n_estimators': 500,
        'random_state': 42
    },
    {
        'name': 'RF_LimitedFeatures',
        'n_estimators': 100,
        'max_features': 0.5,   # use half of the features for each split
        'random_state': 42
    }
]

# 5. Train, evaluate, and collect predictions
results = {}
for cfg in rf_configs:
    name = cfg.pop('name')
    rf   = RandomForestRegressor(**cfg)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"{name}: {cfg} → MSE: {mse:.4f}, R²: {r2:.4f}")
    
    results[name] = preds

# 6. Plot actual vs. predicted for each configuration
plt.figure(figsize=(12, 6))
plt.plot(df_test['Published'], y_test, color='k', linewidth=2, label='Actual Close')

for name, preds in results.items():
    plt.plot(df_test['Published'], preds, label=name)

plt.title("Random Forest: Close Price Prediction Across 5 Configurations")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()
