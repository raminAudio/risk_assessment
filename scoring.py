from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_df = pd.read_csv(test_data_path +'/testdata.csv')
    LM_A = list(test_df['lastmonth_activity'])
    LY_A = list(test_df['lastyear_activity'])
    N_E  = list(test_df['number_of_employees'])
    y  = np.asarray(list(test_df['exited']))
    X = np.asarray([LM_A,LY_A,N_E]).T

    LG = pickle.load(open(model_path + '/trainedmodel.pkl', 'rb'))
    y_predict = LG.predict(X)
    f1_score = metrics.f1_score(y, y_predict)

    f = open(dataset_csv_path +'/latestscore.txt','w')
    f.write(str(f1_score))

if __name__ == '__main__':
    score_model()
