from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# загрузка
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    features = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    cluster = None
    error = None

    if request.method == "POST":
        try:
            data = {}

            for f in features:
                data[f] = float(request.form[f])

            df = pd.DataFrame([data])
            scaled = scaler.transform(df)

            cluster = model.predict(scaled)[0]

        except:
            error = "Enter valid numbers"

    return render_template(
        "index.html",
        features=features,
        cluster=cluster,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)