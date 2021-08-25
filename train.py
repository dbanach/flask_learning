import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report


def save_model(my_model,model_filename):
    """
    Receives a machine learning model and a file name and saves it in the disk using pickle

    """

    with open(model_filename, 'wb') as handle:
        pickle.dump(my_model, handle)

def load_and_get_sets(filename,target):
    """
    function that gets a filename and the name of target feature
    then loads the data from the file and returns the train and test sets
    """


    df = pd.read_csv(filename)
    X = df.loc[:,df.columns!=target]
    y = df.churned


    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=334)
    return X_train,X_test,y_train,y_test

def pred_and_model(X_train,y_train,X_test):
    """

    function that receives data, trains a model and returns the prediction and the fitted model
    """


    rf_clf = RandomForestClassifier(n_estimators=200)
    rf_clf.fit(X_train,y_train)

    y_pred = rf_clf.predict(X_test)

    return y_pred, rf_clf

def main():
    """
    Main function that calls the other functions and controls the flow of the process required


    :return:
    """
    X_train, X_test, y_train, y_test = load_and_get_sets('cellular_churn_greece.csv','churned')
    y_pred, rf_clf = pred_and_model(X_train,y_train,X_test)

    print('Classification report:\n')
    print(classification_report(y_test, y_pred))

    save_model(rf_clf, 'churn_model.pkl')
    print('M.L model saved as churn_model.pkl')
    X_test.to_csv('X_test.csv', index=False)
    print('X_test set saved as X_test.csv')
    np.savetxt('preds.csv',y_pred,delimiter=",")
    print('Model predictions on test set saved as preds.csv')


if __name__ == '__main__':
    main()


