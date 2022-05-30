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
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
model_path       = os.path.join(config['output_model_path'])

def prep_data(df, scaler=None):
    LM_A = list(df['lastmonth_activity'])
    LY_A = list(df['lastyear_activity'])
    N_E  = list(df['number_of_employees'])

    X = np.asarray([LM_A,LY_A,N_E]).T
    if scaler:
        X = scaler.transform(X)
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        sc_file = model_path + '/scaler.sav'
        pickle.dump(scaler, open(sc_file, 'wb'))

    X = scaler.transform(X)
    y  = np.asarray(list(df['exited']))
    return X,y
