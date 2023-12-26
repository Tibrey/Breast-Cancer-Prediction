from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

from logistic import LogisticRegression
from SVM import SVM

# from Logistic_Regression.ipynb import LogisticRegression

app = Flask(__name__, static_url_path="/static")

# Load the trained model from the .pkl file
with open("../model.pkl", "rb") as file:
    model = LogisticRegression.load_model("../model.pkl")

# with open("../SVM_model.pkl", "rb") as file:
#     model = SVM.load_model("../SVM_model.pkl")

# print(model)


# model = pickle.load(open("../model.pkl","rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():

    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        print(request)
        print(features)
        final_features = [np.array(features)]
        mean = np.mean(final_features)
        std = np.std(final_features)
        final_features = (final_features - mean) / std
        output = model.predict(final_features)
        print(output)

       
        if output == 1:
            res_val = "Benign"
        else:
            res_val = "Malignant"

        return render_template(
            "predict.html", prediction="Patient has {}".format(res_val)
        )

    elif request.method == "GET":
        return render_template("predict.html")


@app.route("/data")
def data():
    return render_template("data.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
