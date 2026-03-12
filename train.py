import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import pickle

data = pd.read_csv("data.csv")

X = data[['temperature', 'humidity', 'rainfall', 'wind_speed', 'heat_index']]
y = data['flood_risk']

clf_model = RandomForestClassifier()
clf_model.fit(X, y)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

pickle.dump(clf_model, open("model.pkl", "wb"))
pickle.dump(kmeans, open("kmeans.pkl", "wb"))

print("Classification + Clustering Models saved successfully!")
