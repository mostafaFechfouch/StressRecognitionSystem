# This is a simple script to post a dataframe row in json format to test the prediction API (i installed postman now to use it instead)
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import time
import serial
import subprocess
import sys
import requests
import json

url = 'http://127.0.0.1:8000/predictfromjson'

datastressedETC = {"ECG": {"109545": 0.071}, "EMG": {"109545": 0.821}, "HR": {"109545": 91.0}, "RESP": {"109545": 44.44}, "RRinterval": {"109545": 0.96774194}, "NNRR": {
    "109545": 0.971429}, "RMSSD": {"109545": 0.031294918}, "SDNN": {"109545": 0.0478281587}, "AVNN": {"109545": 0.9843317977}, "pNN50": {"109545": 0.2285714286}}

dataNotStressedETC = {"ECG": {"125871": 0.047}, "EMG": {"125871": 0.942625}, "HR": {"125871": 86.0}, "RESP": {"125871": 33.85}, "RRinterval": {"125871": 0.70967742}, "NNRR": {
    "125871": 0.97619}, "RMSSD": {"125871": 0.0318622455}, "SDNN": {"125871": 0.0336337012}, "AVNN": {"125871": 0.70046083}, "pNN50": {"125871": 0.1666666667}}

r = requests.post(url, json=dataNotStressedETC)

print(r.text)
matplotlib.use("tkAgg")


