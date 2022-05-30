import requests
import json
import os
import ast
#Specify a URL that resolves to your workspace

def call(datafile = 'finaldata',ext = ''):
    URL = "http://127.0.0.1/8000"

    with open('config.json','r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])

    #Call each API endpoint and store the responses
    response0 = requests.get('http://127.0.0.1:8000/prediction?filename=' +datafile+ '.csv').content#put an API call here
    response1 = requests.get('http://127.0.0.1:8000/scoring').content#put an API call here
    response2 = requests.get('http://127.0.0.1:8000/summarystats').content#put an API call here
    response3 = requests.get('http://127.0.0.1:8000/diagnostics').content#put an API call here
    response30 = ast.literal_eval(response3.decode())[0]
    response31 = ast.literal_eval(response3.decode())[2]
    response32 = ' '.join(ast.literal_eval(response3.decode())[3:])

    #combine all API responses
    responses = '\n'.join([ 'http://127.0.0.1:8000/prediction?filename=' +datafile+'.csv', response0.decode(),
                            'http://127.0.0.1:8000/scoring', response1.decode(),
                            'http://127.0.0.1:8000/summarystats', response2.decode(),
                            'http://127.0.0.1:8000/diagnostics', 'summary', response30,
                            'http://127.0.0.1:8000/diagnostics', 'timing', response31,
                            'http://127.0.0.1:8000/diagnostics', 'outdated', response32])#combine reponses here

    #write the responses to your workspace
    f = open(model_path + '/apireturns' + ext + '.txt','w')
    f.write(responses)
    f.close()

if __name__ == "__main__":
    call(datafile= 'finaldata2',ext='')
