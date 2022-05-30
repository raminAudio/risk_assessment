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


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    mdl_pkl = model_path + '/trainedmodel.sav'
    sc_file = model_path + '/scaler.sav'
    scaler = pickle.load(open(sc_file,'rb'))
    model = pickle.load(open(mdl_pkl,'rb'))
    ing_txt = output_folder_path + '/ingestedfiles.txt'
    lsc_txt = model_path + '/latestscore.txt'
    print("Deploying from model trained from data from {} with the score file saved in {}".format(ing_txt,lsc_txt))
    subprocess.call(['cp'] + [ing_txt] + [ 'production_deployment/'])
    subprocess.call(['cp'] + [lsc_txt] + [ 'production_deployment/'])
    pickle.dump(model , open('models/trainedmodel.sav','wb'))
    pickle.dump(scaler, open('models/scaler.sav', 'wb'))


if __name__ == '__main__':
    store_model_into_pickle()
