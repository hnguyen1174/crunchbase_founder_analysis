import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC

if __name__ == "__main__":

    df_main = pd.read_csv('../data/df_processed.csv')

    # Linear regression to check correlation
    X = df_main.drop(['Success_all'], axis = 1)
    y = df_main['Success_all']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=420)

    parameters = {
        'C': (0.001, 0.01, 0.1, 1, 10),
        'gamma': (0.001, 0.01, 0.1, 1)
    }

    svc_search = GridSearchCV(SVC(), parameters, cv=5)
    svc_search.fit(X_train, np.ravel(y_train))