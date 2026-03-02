import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import pickle

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['temperature', 'humidity', 'rainfall', 'wind_speed', 'heat_index']]
y = data['flood_risk']

# Train Classification Model
clf_model = RandomForestClassifier()
clf_model.fit(X, y)

# Train Clustering Model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Save both models
pickle.dump(clf_model, open("model.pkl", "wb"))
pickle.dump(kmeans, open("kmeans.pkl", "wb"))

print("Classification + Clustering Models saved successfully!")
