import joblib
import pandas as pd

# from diabetes.feature_engineering.data_cleansing import perform_data_cleansing
from flask import Flask, jsonify, request

app = Flask(__name__)
model = " "


@app.route("/predict", methods=["POST"])
def predict():
    input_json = request.json
    input_df = pd.DataFrame(input_json)
    # input_df = perform_data_cleansing(input_df)
    prediction = model.predict(input_df)
    return jsonify({"prediction": list(prediction)})


if __name__ == "__main__":
    model = joblib.load("model.pkl")
    app.run(port=8080)
