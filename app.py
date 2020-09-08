#This is the flask api it loads the pickle saved model and get the features in input and output the prediction either stressed or not stressed

#Install Libraries
from flask import Flask, request, jsonify, render_template
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)
d=[9.26000000e+01, 7.07651163e-01, 7.06833000e-01, 2.61184000e-02,
  2.43902000e-02, 3.16474000e-02, 3.16474000e-02, 0.00000000e+00,
  0.00000000e+00, 0.00000000e+00, 3.55569503e+00]

knn = joblib.load("knn_model.pkl") 
#print ('Model loaded: ', knn.predict([d]))

@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form['HR'],
    request.form['IIS'],
    request.form['AVNN'],
    request.form['RMSSD'],
    request.form['pNN50'],
    request.form['TP'],
    request.form['ULF'],
    request.form['VLF'],
    request.form['LF'],
    request.form['HF'],
    request.form['LF_HF']]

    data = np.array([np.asarray(data, dtype=float)])

    predictions = knn.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    if predictions[0]==1:
        class_ = 'stressed'
    else:
        class_ = 'not stressed'


    return render_template('index.html', pred=class_)

@app.route('/predictfromjson', methods=['POST'])
def predictfromjson():
    data= request.get_json()
    df=pd.json_normalize(data)
    #print(df)
    predictions = knn.predict(df)
    print('INFO Predictions: {}'.format(predictions))
    if predictions[0]==1:
        class_ = 'stressed'
    else:
        class_ = 'not stressed'
    print(class_)
    return jsonify({'Prediction':class_})


def main():
    """Run the app."""
    app.run(port=8000, debug=True)


if __name__ == '__main__':
    main()