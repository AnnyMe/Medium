# grasp input from users

import pickle
from flask import Flask, request
import numpy as np
import pandas as pd

# rb specify read binary to read it as binary file instead of normal text file
with open("regression.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# declare a flask app
app = Flask(__name__)


# When a client (such as a web browser) sends a GET or POST request to this URL, Flask will execute the function decorated by this route.
# In this example, the add function is decorated with @app.route('/hello_world', methods=['GET', 'POST']). When the client requests the /hello_world URL
# using a GET or POST request, the add function is executed

# Defines a route for the URL /predict, and specifies it should accept both GET and POST requests
@app.route("/predict", methods=["GET", "POST"])
def predict_regression():

    quality = int(request.args.get("OverallQual"))
    livingarea = int(request.args.get("GrLivArea"))

    prediction = model.predict(np.array([[quality, livingarea]]))

    # prediction = 1

    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_regression_file():
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    # port 5000 is the default port, you can aso set it to port=7000
    app.run(host="0.0.0.0", port=5000)
