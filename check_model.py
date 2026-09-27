import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")

print("Model:", model)
print("Scaler:", scaler)
print("Threshold:", threshold)

print("\nExpected features:")
print(scaler.feature_names_in_)