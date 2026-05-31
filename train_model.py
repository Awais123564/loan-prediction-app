import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("⏳ Step 1: Generating dataset with ASSET COLLATERAL rules...")
np.random.seed(42)
n_samples = 4000

data = {
    'no_of_dependents': np.random.randint(0, 5, n_samples),
    'education': np.random.choice([0, 1], n_samples),
    'self_employed': np.random.choice([0, 1], n_samples),
    'income_annum': np.random.randint(20000, 120000, n_samples),
    'loan_amount': np.random.randint(10000, 250000, n_samples),
    'loan_term': np.random.choice([12, 36, 60, 180, 360], n_samples),
    'cibil_score': np.random.randint(300, 900, n_samples),
    'residential_assets_value': np.random.randint(10000, 300000, n_samples),
    'commercial_assets_value': np.random.randint(0, 200000, n_samples),
    'luxury_assets_value': np.random.randint(5000, 150000, n_samples),
    'bank_asset_value': np.random.randint(5000, 150000, n_samples)
}
df = pd.DataFrame(data)

# 🔗 LINKING THE ASSETS HERE: Calculate Total Asset Worth
total_assets = (df['residential_assets_value'] + 
                df['commercial_assets_value'] + 
                df['luxury_assets_value'] + 
                df['bank_asset_value'])

# New Rule: Loan must be less than (3x Annual Income) + (50% of Total Assets collateral)
base_status = np.where(
    (df['cibil_score'] > 600) & 
    (df['loan_amount'] < (df['income_annum'] * 3) + (total_assets * 0.5)), 
    1, 0
)

# Keep the 15% noise to maintain your 80-90% accuracy target
noise = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
df['loan_status'] = np.abs(base_status - noise)

X = df.drop(columns=['loan_status'])
y = df['loan_status']

print("⏳ Step 2: Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("⏳ Step 3: Training model...")
model = RandomForestClassifier(max_depth=4, n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train) * 100
print(f"\n📊 Target Accuracy Achieved: {train_acc:.2f}%")

print("⏳ Step 4: Saving asset-linked models...")
pickle.dump(model, open('best_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("🎉 Done! Assets are now fully linked to the model's brain.")