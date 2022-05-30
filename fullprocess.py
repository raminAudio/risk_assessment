import os
import pandas as pd
import json


configs = {
  "input_folder_path": "sourcedata",
  "output_folder_path": "ingesteddata",
  "test_data_path": "testdata",
  "output_model_path": "models",
  "prod_deployment_path": "production_deployment"
}

with open("config.json", "w") as write_file:
    json.dump(configs, write_file, indent=4)


with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']


##################Check and read new data
#first, read ingestedfiles.txt
f = open(prod_deployment_path + '/' + 'ingestedfiles.txt' , 'r')
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
files_in_text_file = [f.replace('\n','').split('/')[-1] for f in f.readlines() if '.csv' in f]
f.close()

test_files = open(output_folder_path + '/test_files.txt','w')
files_in_source = [f for f in os.listdir(input_folder_path) if '.csv' in f]
not_listed_files = []
for f in files_in_source:
    if f not in files_in_text_file:
        not_listed_files.append(input_folder_path + '/' +f)
        test_files.write(input_folder_path + '/' +f)
        test_files.write('\n')
        
test_files.close()
if len(not_listed_files)==0:
    print("No new data is found.")
##################Deciding whether to proceed, part 1

GO_NEXT = 0
if len(not_listed_files):
    import ingestion
    ingestion.merge_multiple_dataframe(filename_input = 'test_files' , filename_output = 'testdata_new', testing=1)
    try:
        ingestion.ingestion('a')
        ingestion.merge_multiple_dataframe()
    except:
        ingestion.ingestion('a')
        ingestion.merge_multiple_dataframe()
    GO_NEXT = 1
# ##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
old_score = open(prod_deployment_path + '/' + 'latestscore.txt' , 'r').readline()
new_score = old_score
test_file_new = open(output_folder_path + '/test_data_versions.txt','r').readlines()[-1].replace('\n','').split('/')[1].split('.')[0]

import scoring
if GO_NEXT:
    new_score = scoring.score_model(test_file_new)
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_score < old_score:
    print("Model Drift Detected score on newly ingested data {} and the old score was {}".format(new_score,old_score))
    import training
    training.train_model(ext='2')
    score = scoring.score_model()
    ##################Re-deployment
    # re-run the deployment.py script
    import deployment
    deployment.store_model_into_pickle()
    import diagnostics
    diagnostics.model_predictions()
    diagnostics.dataframe_summary()
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()
    ################## Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    import reporting
    reporting.score_model(ext='2')

    # Call API
    import apicalls
    apicalls.call(datafile= 'finaldata4',ext='2')
