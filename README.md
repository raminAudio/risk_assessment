You'll complete the project in the Udacity Workspace, which is a Linux virtual machine containing all the computing resources you'll need to complete the project.

Your workspace has eight locations you should be aware of:

/home/workspace, the root directory. When you load your workspace, this is the location that will automatically load. This is also the location of many of your starter files.
/practicedata/. This is a directory that contains some data you can use for practice.
/sourcedata/. This is a directory that contains data that you'll load to train your models.
/ingesteddata/. This is a directory that will contain the compiled datasets after your ingestion script.
/testdata/. This directory contains data you can use for testing your models.
/models/. This is a directory that will contain ML models that you create for production.
/practicemodels/. This is a directory that will contain ML models that you create as practice.
/production_deployment/. This is a directory that will contain your final, deployed models.
Important notes about the workspace:

Your files under the /home/workspace/ directory are saved for you automatically. However, after a 30 minutes idle time (navigate away from the Workspace, tab not accessed, or simply closed, or laptop asleep, etc), your Workspace will go sleep. If you return after this period, your files will be restored to your most recent work, but you will lose the list of open files or any shell sessions you may have had running.

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
