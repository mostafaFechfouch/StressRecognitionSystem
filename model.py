#This is a KNN model with K=5 it gave us an accuracy of 0.76 
from sklearn.model_selection import train_test_split
import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from scipy import signal 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Loading the dataset 
dataframe_hrv = pd.read_csv("df.csv")
dataframe_hrv = dataframe_hrv.reset_index(drop=True)

#Stress labels are in float format indecating the level of stress we will convert it to binary format (either stressed 1 or not stressed 0)
def fix_stress_labels(df='',label_column='stress'):
    df['stress'] = np.where(df['stress']>=0.5, 1, 0)
    return df
dataframe_hrv = fix_stress_labels(df=dataframe_hrv)

#Fixing the missing values 
def missing_values(df):
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13) 

    df=df.fillna(df.mean())
    return df
dataframe_hrv = missing_values(dataframe_hrv)

#Feature selection
selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']
X = dataframe_hrv[selected_x_columns]
y = dataframe_hrv['stress']

# spliting dataset 80% for training and 20% for test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
X_train.shape

#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=5)

#Fit the model
knn.fit(X_train,y_train)

#Get accuracy
knn.score(X_test,y_test)

# ---------------- I deprecated this part with the last part using joblib  --------------------

#saving pickling the model to knnpickle_file.pkl
#knnPickle = open('knnpickle_file', 'wb') 

# source, destination 
#pickle.dump(knn, knnPickle)  

# load the model from disk to test it
#loaded_model = pickle.load(open('knnpickle_file', 'rb'))
#result = loaded_model.predict(X_test) 

#Classsification report
#print(classification_report(y_test,result))

# -----------------------------------------------------------------------------------------------

#Serialize the model and save using joblib to knn_model.pkl file
joblib.dump(knn, 'knn_model.pkl')
print("KNN Model Saved")

# Save features from training
knn_columns = list(X_train.columns)
joblib.dump(knn_columns, 'knn_columns.pkl')
print("KNN Colums Saved")

#Load the model
knn = joblib.load('knn_model.pkl')
y_predict = knn.predict(X_test)
print(classification_report(y_test,y_predict))