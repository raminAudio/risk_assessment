from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
from data_prep import *

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
model_path       = os.path.join(config['output_model_path'])


#################Function for training the model
def train_model(ext=''):
    fversion = open(output_folder_path + '/train_data_versions.txt','r').readlines()[-1].replace('\n','')
    print("Reading " + fversion )
    df = pd.read_csv(fversion)
    X,y = prep_data(df)

    print("Training input shape {} and output shape {}".format(X.shape,y.shape))
    #use this logistic regression for training
    LG = LogisticRegression(C=1.0, solver='liblinear', max_iter=500).fit(X,y)
    #fit the logistic regression to your data
    #write the trained model to your workspace in a file called trainedmodel.pkl
    model_file = model_path + '/trainedmodel' + ext + '.sav'
    print("Saving " + model_file)
    pickle.dump(LG, open(model_file, 'wb'))

if __name__ == '__main__':
    train_model()
