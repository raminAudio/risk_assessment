from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import subprocess
from sklearn.preprocessing import Normalizer
from data_prep import *



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model(filename = 'testdata'):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    print("Testing on " + filename)
    test_df = pd.read_csv(test_data_path +'/' + filename + '.csv')

    sc_file = model_path + '/scaler.sav'
    scaler = pickle.load(open(sc_file, 'rb'))
    X,y = prep_data(test_df, scaler = scaler)

    model_file = model_path + '/trainedmodel.sav'
    print("Loading " + model_file)

    LG = pickle.load(open(model_file, 'rb'))
    y_predict = LG.predict(X)

    f1_score = metrics.f1_score(y, y_predict)
    score_filename = model_path +'/latestscore.txt'

    print("Saving score to " + score_filename)
    f = open(score_filename,'w')
    f.write(str(f1_score))
    print("f1score {}".format(f1_score))
    return str(f1_score)

if __name__ == '__main__':
    score_model()
