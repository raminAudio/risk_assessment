import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from data_prep import *

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model(trainmodel_filename='trainedmodel', test_filename = 'testdata',ext =''):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_df = pd.read_csv(test_data_path +'/' + test_filename + '.csv')
    sc_file = model_path + '/scaler.sav'
    scaler = pickle.load(open(sc_file, 'rb'))
    X,y = prep_data(test_df, scaler)

    model_filename = model_path + '/' + trainmodel_filename+ '.sav'
    model = pickle.load(open(model_filename,'rb'))

    y_predict = model.predict(X)
    cm = metrics.confusion_matrix(y,y_predict)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
    disp.plot()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(model_path + '/confusionmatrix' + ext + '.png')


if __name__ == '__main__':
    score_model()
