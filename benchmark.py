import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import time

# 1. Load data
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# 2. Prepare data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
print("Training LightGBM model on CPU...")
start_time = time.time()
model = lgb.LGBMClassifier(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)
end_time = time.time()

# 4. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\n--- Results for n2-standard-8 ---")
print(f"Training Time: {end_time - start_time:.4f} seconds")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
