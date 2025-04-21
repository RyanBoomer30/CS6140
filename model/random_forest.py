import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the merged dataset and parse dates
df = pd.read_csv('data/final_dataset.csv', parse_dates=['Published'])

# 2. Split into train/test by date
cutoff = pd.Timestamp('2025-01-01')
df_train = df[df['Published'] < cutoff]
df_test  = df[df['Published'] >= cutoff]

# 3. Define feature columns (everything except Published and Close)
features = [
    'Sentiment Score', 'High', 'Low', 'Open', 'Volume',
    'score_sq', 'score_cu', 'score_dev',
    'inverted_score', 'skewed_score', 'inverted_skewed_score'
]

# 4. Extract X and y
X_train = df_train[features]
y_train = df_train['Close']
X_test  = df_test[features]
y_test  = df_test['Close']

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

# 5. Train the Random Forest on all features
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Predict & evaluate
preds = rf.predict(X_test)
mse  = mean_squared_error(y_test, preds)
r2   = r2_score(y_test, preds)
print(f"RF (all features) → MSE: {mse:.4f}, R²: {r2:.4f}")

# 7. Plot actual vs predicted over time
plt.figure(figsize=(10, 4))
plt.plot(df_test['Published'], y_test,    label='Actual Close', linewidth=2)
plt.plot(df_test['Published'], preds,     label='Predicted Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Random Forest: Close Price Prediction Using All Features')
plt.legend()
plt.tight_layout()
plt.show()
