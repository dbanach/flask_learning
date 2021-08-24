import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request
app = Flask(__name__)

def get_model(filename):
    """
    Function that loads a file using pickle and returns the M.L model
    """
    with open(filename,'rb') as handle:
        clf = pickle.load(handle)

    return clf

@app.route('/predict_churn')
def single_predict():
    ismale = request.args.get('ismale')
    num_inters = request.args.get('num_inters')
    late_on_payment = request.args.get('late_on_payment')
    age = request.args.get('age')
    years_in_contract = request.args.get('years_in_contract')


    return "What pill will you take "+request.args.get('name') + '?'




def main():
    clf = get_model('churn_model.pkl')
    X_test = pd.read_csv('X_test.csv')
    preds = np.loadtxt('preds.csv')
    y_pred = clf.predict(X_test)

    assert (y_pred == preds).all()
    #if there's no error, it means they are the same.



if __name__ == '__main__':
    main()