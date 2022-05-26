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

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    ing_txt = dataset_csv_path + '/ingestedfiles.txt'
    lsc_txt = dataset_csv_path + '/latestscore.txt'
    subprocess.call(['cp'] + [ing_txt] + [ 'production_deployment/'])
    subprocess.call(['cp'] + [lsc_txt] + [ 'production_deployment/'])
    pickle.dump(model, open('production_deployment/trainedmodel.pkl','wb'))


if __name__ == '__main__':
    model_path = os.path.join(config['output_model_path'])
    mdl_pkl = model_path + '/trainedmodel.pkl'
    model = pickle.load(open(mdl_pkl,'rb'))
    store_model_into_pickle(model)
