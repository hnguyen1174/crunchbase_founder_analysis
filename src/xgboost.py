import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

if __name__ == "__main__":

    df_main = pd.read_csv('../data/df_processed.csv')

    # Linear regression to check correlation
    X = df_main.drop(['Success_all'], axis = 1)
    y = df_main['Success_all']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=420)
    
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=420)

    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  
        "max_depth": randint(2, 6),
        "n_estimators": randint(100, 150),
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, 
                                random_state=420, n_iter=100, 
                                cv=3, verbose=1, n_jobs=1, return_train_score=True)

    search.fit(X_train, y_train)

    report_best_scores(search.cv_results_, 1)