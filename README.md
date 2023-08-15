In order to make sure the files are not overwritten, I used extensions at the end of the files and keep track of files ingested in two text files, ingestedfiles.txt and test_files.txt . The final concatenated files are kept track in train_data_version.txt and test_data_versions.txt . 

First,

  run firstRun.py
  
    -- this will ingest existing data then concatenate dataframes --> 
    
    from practicedata/ and save it in ingesteddata/finaldata2.csv 
    
    -- run training.py a logistic regression model -->
    
    practicemodels/trainedmodel.sav and practicemodels/scaler.sav  and ingesteddata/train_data_version.txt 
    
    -- run scoring.py the model on test dataset -->
    
    practicemodels/latestscore.txt \n
    
    -- run deployment.py it --> 
    
    copy data from practicemodels/ to production_deployment and models \n
    
    -- run diagnostics.py --> 
    
    reads in from ingesteddata/train_data_version.txt and statistics from practicemodels/
    
    -- run reporting.py --> 
    
    saves practicemodels/confusionmatrix.png using the model from practicemodels/

Second,

  run fullprocess.py
  
    -- Reads in the existing ingested files text files, and check to see if there are any new files added.
    
    -- If no new data is added (from sourcedata/), it will ingest them using ingestion.py --> 
    
    this will ingesteddata/finaldata4.csv as well as a new test dataset called testdata/testdata_new.csv and update test_data_versions.txt and train_data_version.txt
    
    -- It will then read the latestscore.txt from last step and if the score on the new test dataset is less than the score from last step, it will train a new model on the new ingested training data, ingesteddata/finaldata4.csv and redploy it.
    
    -- It will then diagnostics.py and reporting.py as well as apicalls.py (app.py should have been ran before hand.)



/practicedata/. This is a directory that contains some data you can use for practice.

/sourcedata/. This is a directory that contains data that you'll load to train your models.

/ingesteddata/. This is a directory that will contain the compiled datasets after your ingestion script.

/testdata/. This directory contains data you can use for testing your models.

/models/. This is a directory that will contain ML models that you create for production.

/practicemodels/. This is a directory that will contain ML models that you create as practice.

/production_deployment/. This is a directory that will contain your final, deployed models.

Starter Files
There are many files in the starter: 10 Python scripts, one configuration file, one requirements file, and five datasets.

The following are the Python files that are in the starter files:

training.py, a Python script meant to train an ML model

scoring.py, a Python script meant to score an ML model

deployment.py, a Python script meant to deploy a trained ML model

ingestion.py, a Python script meant to ingest new data

diagnostics.py, a Python script meant to measure model and data diagnostics

reporting.py, a Python script meant to generate reports about model metrics

app.py, a Python script meant to contain API endpoints

wsgi.py, a Python script to help with API deployment

apicalls.py, a Python script meant to call your API endpoints

fullprocess.py, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed

The following are the datasets that are included in your starter files. Each of them is fabricated datasets that have information about hypothetical corporations.

Note: these data have been uploaded to your workspace as well

dataset1.csv and dataset2.csv, found in /practicedata/

dataset3.csv and dataset4.csv, found in /sourcedata/

testdata.csv, found in /testdata/

The following are other files that are included in your starter files:

requirements.txt, a text file and records the current versions of all the modules that your scripts use

config.json, a data file that contains names of files that will be used for configuration of your ML Python scripts
