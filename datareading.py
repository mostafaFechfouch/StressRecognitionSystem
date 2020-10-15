import serial
import time
import csv
import numpy as np
import pandas as pd
import datetime

ser = serial.Serial('COM3',19200)
ser.flushInput()

i=0
ts = time.gmtime()
hours=time.strftime("%H", ts)
minutes=time.strftime("%M", ts)
seconds=time.strftime("%S", ts)
initialtime=int(hours)*60+int(minutes)*60+int(seconds)
print("initialtime: ",initialtime)
while True:
    ts = time.gmtime()
    currenttime=(int(time.strftime("%H", ts))*60)+(int(time.strftime("%M", ts))*60)+(int(time.strftime("%S", ts)))-initialtime
    print("---------------------------------")
    print(currenttime)
    i=0
    data=[]
    while i<4:
        try:
            ser_bytes = ser.readline()
            try:
                decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
                data.append(decoded_bytes)
            except:
                continue
        except:
            print("Keyboard Interrupt")
            break
        i=i+1
    with open('data.csv', 'a',newline='') as csvfile:
        fieldnames = ['time', 'ECG','EMG','HR','RESP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'time': currenttime, 'ECG': data[0],'EMG':data[1],'HR':data[2],'RESP':data[3]})