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


files = [input_folder_path+'/' + f for f in os.listdir(input_folder_path) if 'csv' in f]
#############Function for data ingestion
def merge_multiple_dataframe():
    ingestF = open(output_folder_path + '/ingestedfiles.txt','w')
    #check for datasets, compile them together, and write to an output file
    list_of_dfs = []
    for f in files:
        ingestF.write(f)
        ingestF.write('\n')
        df = pd.read_csv(f)
        list_of_dfs.append(df)
    df_merged = pd.concat(list_of_dfs)
    df_merged.drop_duplicates(inplace=True)
    df_merged.to_csv(output_folder_path + '/finaldata.csv')
    ingestF.close()
    return df_merged

if __name__ == '__main__':
    merge_multiple_dataframe()
