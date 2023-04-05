from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from flask import *

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/forest_fire")
def forest_fire():
    return render_template("forest_fire.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][0], 2)

    if output > str(0.5):
        return render_template('forest_fire.html',
                               pred='This area is in Danger.\nProbability of crime occuring is {}'.format(output),
                               bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',
                               pred='This area is safe.\n Probability of crime occuring is {}'.format(output),
                               bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
