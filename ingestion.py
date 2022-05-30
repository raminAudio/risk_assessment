import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
test_data_path = config['test_data_path']

# Existing files
def ingestion(WRITE_APPEND = 'w'):
    files_input_folder = [input_folder_path+'/' + f for f in os.listdir(input_folder_path) if 'csv' in f]
    try:
        files_ingested = [f.replace('\n','') for f in open(output_folder_path + '/ingestedfiles.txt','r').readlines()]
    except:
        files_ingested = []
    ingestF = open(output_folder_path + '/ingestedfiles.txt',WRITE_APPEND)
    for f in files_input_folder:
        if f not in files_ingested:
            ingestF.write(f)
            ingestF.write('\n')
    ingestF.close()

#############Function for data ingestion
def merge_multiple_dataframe(filename_input = 'ingestedfiles' , filename_output = 'finaldata', testing=0):
    if not testing:
        ftrainversion = open(output_folder_path + '/train_data_versions.txt','w')
    else:
        ftestversion = open(output_folder_path + '/test_data_versions.txt','w')

    ingestF = open(output_folder_path + '/' + filename_input + '.txt','r')
    all_files = ingestF.readlines()
    #check for datasets, compile them together, and write to an output file
    list_of_dfs = []
    for f in all_files:
        print(f)
        f = f.replace('\n','')
        df = pd.read_csv(f)
        list_of_dfs.append(df)
    df_merged = pd.concat(list_of_dfs).reset_index().drop('index',axis=1)
    df_merged.drop_duplicates(inplace=True)

    if not testing:
        latest_datafile = output_folder_path + '/' + filename_output + str(len(all_files)) + '.csv'
        print("Saving " + latest_datafile )
        ftrainversion.write(latest_datafile)
        ftrainversion.write('\n')
    else:
        latest_datafile = test_data_path + '/' + filename_output + '.csv'
        print("Saving " + latest_datafile )
        ftestversion.write(latest_datafile)
        ftestversion.write('\n')

    df_merged.to_csv(latest_datafile)
    ingestF.close()
    return df_merged

if __name__ == '__main__':
    ingestion()
    merge_multiple_dataframe()
