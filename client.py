import pickle
import pandas as pd
import numpy as np
import requests

def ask_and_get_predictions(df,rows=5):
    """
    Function that gets the data and the numberd of rows to ask the webpage for predictions and returns them
    as a numpy array
    """


    predictions = []
    for index in range(rows):

        payload = get_payload(df.loc[index,:],df.loc[index,:].index)
        r = requests.get('http://localhost:5000/predict_churn', params=payload)
        predictions.append(float(r.text))

    return np.array(predictions)

def get_payload(values, indexes):
    """
    Function that gets values and indexes and returns a dictionary with that information.

    """

    my_dict = {}
    for i,j in zip(values,indexes):
        my_dict[j] = i
    return my_dict

if __name__ == '__main__':

    X_test = pd.read_csv('X_test.csv')
    preds = np.loadtxt('preds.csv')
    y_pred = ask_and_get_predictions(X_test)

    #now we'll check that the output is the same

    print('Is our result the same as what was on the preds.csv file? ')
    print((y_pred==preds[:5]).all())

    print('Predictions from file preds.csv:\n')
    print(preds[:5])

    print('\nPredictions received from the webpage:\n')
    print(y_pred)
