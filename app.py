# This is the flask api it loads the pickle saved model and get the features in input and output the prediction either stressed or not stressed

# Import Libraries
from flask import Flask, request, jsonify, render_template
import joblib
import traceback
import pandas as pd
import numpy as np
import serial
import time
import subprocess
import os
from ecgdetectors import Detectors
import scipy.signal as signal

app = Flask(__name__)

ExtraTreesClassifier = joblib.load("static/models/ExtraTreesClassifier.pkl")
print('Model loaded')
@app.route("/")
def index():
    return render_template('home.html')


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

    predictions = ExtraTreesClassifier.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    if predictions[0] == 1:
        class_ = 'stressed'
    else:
        class_ = 'not stressed'

    return render_template('index.html', pred=class_)


@app.route('/predictfromjson', methods=['POST'])
def predictfromjson():
    data = request.get_json()
    df = pd.json_normalize(data)
    # print(df)
    predictions = ExtraTreesClassifier.predict(df)
    print('INFO Predictions: {}'.format(predictions))
    if predictions[0] == 1:
        class_ = 'stressed'
    else:
        class_ = 'not stressed'
    print(class_)
    return jsonify({'Prediction': class_})


@app.route("/home")
def home():
    return render_template('home.html')


sp = 0


@app.route("/startmeasuring", methods=['POST'])
def startmeasuring():
    proc = subprocess.Popen(["python", "datareading.py"], shell=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    global sp
    sp = proc.pid
    print("subprocess started ", sp)
    print("subprocess started with id ", proc.pid)
    return str(proc.pid)


@app.route("/stopmeasuring", methods=['POST'])
def stopmeasuring():
    global sp
    print('killing process with id: ', sp)
    os.system("taskkill /f /t /pid "+str(sp))
    print("subprocess stopped")
    df = pd.read_csv('data.csv', index_col=None, header=0)
    ecg_signal = df['ECG']*1000
    sr = 60
    time = np.linspace(0, len(ecg_signal) / sr, len(ecg_signal))
    detectors = Detectors(sr)
    r_peaks = detectors.pan_tompkins_detector(ecg_signal)
    peaks = np.array(time)[r_peaks]
    RRinterval = np.diff(peaks)
    median=np.median(RRinterval)
    RRinterval=np.append(RRinterval,median)
    df=df.iloc[r_peaks]
    df["RRinterval"]=RRinterval
    df["RRinterval"] = np.where(df["RRinterval"]<0.5, median,df["RRinterval"])
    df["RRinterval"] = np.where(df["RRinterval"]>1.5, median,df["RRinterval"])
    df["RRinterval"]=signal.medfilt(df["RRinterval"], 5)
    rri=df['RRinterval']
    diff_rri = np.diff(rri)
    NNRR=round(len(np.diff(rri))/len(rri),6)
    RMSSD=np.sqrt(np.mean(diff_rri ** 2))
    SDNN = np.nanstd(rri, ddof=1)
    AVNN = np.nanmean(rri)
    nn50 = np.sum(np.abs(diff_rri) > 0.05)
    pNN50 = nn50 / len(rri)
    df['NNRR']=NNRR
    df['RMSSD']=RMSSD
    df['SDNN']=SDNN
    df['AVNN']=AVNN
    df['pNN50']=pNN50
    print(df.describe())
    df = df.replace([np.inf, -np.inf], np.nan)
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],5)
    df['RESP'].fillna((df['RESP'].mean()), inplace=True)
    df['RESP'] = signal.medfilt(df['RESP'],5)
    df['EMG'].fillna((df['EMG'].mean()), inplace=True)
    df['EMG'] = signal.medfilt(df['EMG'],5)
    df['ECG'].fillna((df['ECG'].mean()), inplace=True)
    df['ECG'] = signal.medfilt(df['ECG'],5) 
    df=df.fillna(df.mean())
    df.to_csv('test_data2.csv', index=False)
    df=df.drop(['time'], axis=1)
    predictions = ExtraTreesClassifier.predict(df)
    print('predictions: ',predictions)
    c=(predictions == 0).sum()
    c1=(predictions == 1).sum()
    if c<c1:
        class_ = 'stressed'
        return render_template('stressed.html', pred=class_)
    else:
        class_ = 'not stressed'
        return render_template('notstressed.html', pred=class_)
    print(class_)


@app.route("/chart")
def chart():
    return render_template('data.html')


@app.route("/data")
def data():
    data = pd.read_csv('data.csv', index_col=None, header=0)
    if data.empty:
        return str(0) 
    else:
        return str(data.iloc[-1][1])

@app.route("/ecgdata")
def ecgdata():
    data = pd.read_csv('data.csv', index_col=None, header=0)
    if data.empty:
        return str(0) 
    else:
        return str(data.iloc[-1][1])

@app.route("/emgdata")
def emgdata():
    data = pd.read_csv('data.csv', index_col=None, header=0)
    if data.empty:
        return str(0) 
    else:
        return str(data.iloc[-1][2])

@app.route("/hrdata")
def hrdata():
    data = pd.read_csv('data.csv', index_col=None, header=0)
    if data.empty:
        return str(0) 
    else:
        return str(data.iloc[-1][3])

@app.route("/respdata")
def respdata():
    data = pd.read_csv('data.csv', index_col=None, header=0)
    if data.empty:
        return str(0) 
    else:
        return str(data.iloc[-1][4])

def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    main()
