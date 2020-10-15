from sklearn.model_selection import train_test_split
import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from scipy import signal 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import glob
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

df=pd.read_csv('../data/joinedcleaned.csv')
#print(df.describe())
def missing_values(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13)
    df['hand GSR'].fillna((df['hand GSR'].mean()), inplace=True)
    df['hand GSR'] = signal.medfilt(df['hand GSR'],13)
    df['EMG'].fillna((df['EMG'].mean()), inplace=True)
    df['EMG'] = signal.medfilt(df['EMG'],13)
    df['marker'].fillna((df['marker'].mean()), inplace=True)
    df['marker'] = signal.medfilt(df['marker'],13) 
    df=df.fillna(df.mean())
    return df
df = missing_values(df)

X = df.drop(['stress','annotation','hand GSR','foot GSR','time','marker'], axis=1)
y = df['stress']
X_train, X_test, y_train, y_test = train_test_split(X, y)
etc = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
etc=etc.fit(X_train, y_train)
y_pred=etc.predict(X_test)
accuracy=etc.score(X_test, y_test)
f1score=f1_score(y_pred,y_test)
print('accuracy: ',accuracy)
print('f1score: ', f1score)
print('classification report: ',classification_report(y_test,y_pred))
print('confusion matrix: ',confusion_matrix(y_test,y_pred))
#Serialize the model and save using joblib to knn_model.pkl file
joblib.dump(etc, 'ExtraTreesClassifier.pkl')
print("Extra Trees Classifier Saved")

# Save features from training
etc_columns = list(X_train.columns)
joblib.dump(etc_columns, 'ExtraTreesClassifier_columns.pkl')
print("Extra Trees Classifier Colums Saved")

#Load the model
etc = joblib.load('ExtraTreesClassifier.pkl')
y_predict = etc.predict(X_test)
print('classification report loaded: ',classification_report(y_test,y_predict))
print('confusion matrix loaded: ',confusion_matrix(y_test,y_predict))