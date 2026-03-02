from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
model = pickle.load(open("model.pkl", "rb"))
# Feature Importance (Global)
features = ["Temperature", "Humidity", "Rainfall", "Wind Speed"]
feature_importance = model.feature_importances_
importance_data = list(zip(features, feature_importance))
kmeans = pickle.load(open("kmeans.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
     
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        wind_speed = float(request.form['wind_speed'])


        heat_index = temperature + (0.33 * humidity) - (0.7 * wind_speed) - 4.0
        heat_index = round(heat_index, 2)

        # Heat Severity Label
        if heat_index < 30:
            heat_label = "Normal"
        elif heat_index < 40:
            heat_label = "Caution"
        else:
            heat_label = "Extreme"

        features = np.array([[temperature, humidity, rainfall, wind_speed, heat_index]])


        cluster = int(kmeans.predict(features)[0])

        if cluster == 0:
            cluster_label = "Low Climate Stress Zone"
        elif cluster == 1:
            cluster_label = "Moderate Climate Stress Zone"
        else:
            cluster_label = "High Climate Stress Zone"

        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)

        confidence = round(probability[0][prediction] * 100, 2)

        if prediction == 1:
            risk = "High Flood Risk ⚠️"
            color = "danger"
        else:
            risk = "Low Flood Risk ✅"
            color = "success"

        return render_template('index.html',
            prediction_text=risk,
            confidence=confidence,
            cluster_label=cluster_label,
            heat_index=heat_index,
            heat_label=heat_label,
            importance_data=importance_data,
            color=color
        )

    except Exception:
        return render_template(
            'index.html',
            prediction_text="Invalid Input ❌",
            confidence=0,
            color="secondary"
        )


if __name__ == "__main__":
    app.run(debug=True)
