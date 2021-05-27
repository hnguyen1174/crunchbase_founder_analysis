import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    df_main = pd.read_csv('../data/df_processed.csv')

    # Linear regression to check correlation
    X = df_main.drop(['Success_all'], axis = 1)
    y = df_main['Success_all']

    feature_list = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=420)

    # Simple Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    coef_dict = {}
    for c, f in zip(logreg.coef_[0,:],feature_list):
        coef_dict[f] = c

    y_pred_log = logreg.predict(X_train)
    acc_log_train = logreg.score(X_train, y_train)
    print('Accuracy on the training set is {:.2%}'.format(acc_log_train))