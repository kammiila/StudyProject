from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Загрузка модели и scaler

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Загрузка датасета

df = pd.read_csv("customer_data.csv")

# Первые 10 строк
sample_data = df.head(10).to_dict(orient="records")

# Названия колонок
columns = df.columns.tolist()


# Главная страница

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            user_data = {}

            # Получаем значения из формы
            for feature in feature_names:
                value = request.form.get(feature)

                if value == "":
                    value = 0

                user_data[feature] = float(value)

            # DataFrame
            input_df = pd.DataFrame([user_data])

            # Масштабирование
            input_scaled = scaler.transform(input_df)

            # Предсказание кластера
            prediction = model.predict(input_scaled)[0]

        except ValueError:
            error = "Enter valid numbers only"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        features=feature_names,
        sample_data=sample_data,
        columns=columns
    )


# Запуск

if __name__ == "__main__":
    app.run(debug=True)