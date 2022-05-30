import os
import pandas as pd
import json


configs = {
  "input_folder_path": "practicedata",
  "output_folder_path": "ingesteddata",
  "test_data_path": "testdata",
  "output_model_path": "practicemodels",
  "prod_deployment_path": "production_deployment"
}

with open("config.json", "w") as write_file:
    json.dump(configs, write_file, indent=4)

import ingestion

try:
    # first run
    ingestion.ingestion()
    ingestion.merge_multiple_dataframe()
except:
    #second fun
    ingestion.ingestion()
    ingestion.merge_multiple_dataframe()


import training
training.train_model()

import scoring
new_score = scoring.score_model()

import deployment
deployment.store_model_into_pickle()

import diagnostics
diagnostics.model_predictions()
diagnostics.dataframe_summary()
diagnostics.execution_time()
diagnostics.outdated_packages_list()

import reporting
reporting.score_model()
