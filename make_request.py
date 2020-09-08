#This is a simple script to post a dataframe row in json format to test the prediction API (i installed postman now to use it instead)
import requests
import json

url='http://127.0.0.1:8000/predictfromjson'

dataNotStressed={"HR":{"282":72.48120513},"IIS":{"282":0.814128205},"AVNN":{"282":0.813474},"RMSSD":{"282":0.0318934},"pNN50":{"282":0.027027},"TP":{"282":0.0563325},"ULF":{"282":0.0442237},"VLF":{"282":0.0121088},"LF":{"282":0.0},"HF":{"282":0.0},"LF_HF":{"282":3.5556950345}}

dataStressed= {"HR":{"1716":77.90243902},"interval in seconds":{"1716":0.809875},"AVNN":{"1716":0.809128},"RMSSD":{"1716":0.0147978},"pNN50":{"1716":0.0526316},"TP":{"1716":0.022409},"ULF":{"1716":0.022409},"VLF":{"1716":0.0},"LF":{"1716":0.0},"HF":{"1716":0.0},"LF_HF":{"1716":3.5556950345}}

r=requests.post(url , json= dataStressed)

print(r.text)