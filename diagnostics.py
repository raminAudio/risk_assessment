
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import copy
from data_prep import *
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    LG  = pickle.load(open(model_path + '/trainedmodel.sav','rb'))
    df = pd.read_csv(test_data_path + '/testdata.csv')
    sc_file = model_path + '/scaler.sav'
    scaler = pickle.load(open(sc_file, 'rb'))
    X,y = prep_data(df, scaler)
    y_predict = list(LG.predict(X))
    return y_predict#return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    fversion = open(output_folder_path + '/train_data_versions.txt','r').readlines()[-1].replace('\n','')
    print("Reading " + fversion )
    df = pd.read_csv(fversion)
    df.drop('Unnamed: 0', axis = 1,inplace=True)
    list_statistics = {}
    for col in df.columns:
        nan_percent = round(len(df[col].dropna())/len(df[col]),3)
        values = list(df[col].dropna())

        if type(values[0]) != str:
            mu = round(np.mean(values),3)
            med = round(np.median(values),3)
            std = round(np.std(values),3)
            list_statistics[col] = [mu,med,std,nan_percent]
        else:
            # Counting NAN values 
            nan_percent = 1-(len(df[col].dropna())/len(values))
            list_statistics[col] = [nan_percent]
    return list_statistics#return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    training_time = timeit.timeit("train_model()","from training import train_model",number=1)
    ingestion_time = timeit.timeit("merge_multiple_dataframe()","from ingestion import merge_multiple_dataframe",number=1)
    times = [(ingestion_time),(training_time)]
    return str(times) #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    list_packages = []
    package_df = {'library':'','current':None,'latest':None,'used':None}
    texts = open("requirements.txt",'r').readlines()
    for txt in texts:
        library, version = txt.replace('\n','').split('==')
        package_df['library'] = library
        package_df['current'] = version
        list_packages.append(copy.copy(package_df))

    df = pd.DataFrame(list_packages)
    subprocess.run(" pip list --outdated > requirements_installed.txt", shell=True)
    texts = open("requirements_installed.txt",'r').readlines()
    for txt in texts:
         list_of_items = txt.split()
         try:
             library, version,latest, type = list_of_items[0],list_of_items[1],list_of_items[2],list_of_items[3]
             df.loc[df['library'] == library,'latest'] = latest
             df.loc[df['library'] == library,'used']   = version
         except:
            pass
    df.dropna().to_csv('req_versions.csv')
    return df.dropna()


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
