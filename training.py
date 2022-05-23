from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path       = os.path.join(config['output_model_path'])

#################Function for training the model
def train_model():
    df = pd.read_csv(dataset_csv_path +'/finaldata.csv')
    LM_A = list(df['lastmonth_activity'])
    LY_A = list(df['lastyear_activity'])
    N_E  = list(df['number_of_employees'])
    y  = np.asarray(list(df['exited']))
    X = np.asarray([LM_A,LY_A,N_E]).T
    #use this logistic regression for training
    LG = LogisticRegression()
    #fit the logistic regression to your data
    LG.fit(X,y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(LG, open(model_path + '/trainedmodel.pkl', 'wb'))

if __name__ == '__main__':
    train_model()
