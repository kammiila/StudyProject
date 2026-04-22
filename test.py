import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    features = pickle.load(f)

def get_number(name):
    while True:
        try:
            return float(input(f"{name}: "))
        except:
            print("Enter number")

print("Enter customer data:")

data = {}
for f in features:
    data[f] = get_number(f)

df = pd.DataFrame([data])

scaled = scaler.transform(df)
cluster = model.predict(scaled)

print(f"\nCluster: {cluster[0]}")