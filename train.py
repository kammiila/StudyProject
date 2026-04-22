import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Загрузка данных
df = pd.read_csv("customer_data.csv")

print(df.head())
print(df.columns)

# 2. Удаляем пустые значения
df = df.dropna()

# 3. Кодируем категории (если есть текст)
df = pd.get_dummies(df, drop_first=True)

# 4. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 5. Обучение KMeans
model = KMeans(n_clusters=4, random_state=42)
model.fit(X_scaled)

# 6. Добавляем кластер
df["Cluster"] = model.labels_

# 7. Сохранение
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(df.columns[:-1].tolist(), f)

df.to_csv("final_dataset.csv", index=False)

# 8. График
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["Cluster"])

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Customer Segmentation")

plt.savefig("static/cluster_plot.png")
plt.close()

print("✅ TRAIN DONE")