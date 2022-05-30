from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from scoring import score_model
from diagnostics import *

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

prediction_model = pickle.load(open(model_path + '/trainedmodel.sav','rb'))

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def prediction():
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    df =pd.read_csv(dataset_csv_path + '/' +filename)
    LM_A = list(df['lastmonth_activity'])
    LY_A = list(df['lastyear_activity'])
    N_E  = list(df['number_of_employees'])
    X = np.asarray([LM_A,LY_A,N_E]).T
    predictions = prediction_model.predict(X)
    return str(predictions)#add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    return score_model()#add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    #check means, medians, and modes for each column
    return dataframe_summary()#return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    summary_stats = dataframe_summary()
    timing = execution_time()
    outdated = outdated_packages_list().values
    result = [str(summary_stats),str(timing),str(outdated)]
    return str([result[0],'\n' ,result[1],'\n',result[2]])#add return value for all diagnostics

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
