import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("tyre_dataset.csv")

X = df[['Distance','LoadWeight','Pressure',
        'Temperature','TyreAge']]

y = df['Maintenance']

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "tyre_model.pkl")

print("Model Saved Successfully")
