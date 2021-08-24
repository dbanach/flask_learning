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
    """
    function that receives the data from the user and predicts if will churn and returns
    a string "0" or "1"
    :return:
    """

    ismale = request.args.get('is_male')
    num_inters = request.args.get('num_inters')
    late_on_payment = request.args.get('late_on_payment')
    age = request.args.get('age')
    years_in_contract = request.args.get('years_in_contract')
    data =[ismale,num_inters,late_on_payment,age,years_in_contract]
    for d in data:
        print(d,type(d))


    data = [float(elem) for elem in data]
    data = np.array(data).reshape(1,-1)
    prediction = str(int(clf.predict(data)))


    return prediction




def main():
    clf = get_model('churn_model.pkl')
    X_test = pd.read_csv('X_test.csv')
    preds = np.loadtxt('preds.csv')
    y_pred = clf.predict(X_test)

    assert (y_pred == preds).all()
    #if there's no error, it means they are the same.

    return clf

if __name__ == '__main__':
    clf = main()
    app.run()