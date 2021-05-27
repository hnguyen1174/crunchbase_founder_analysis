import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

    df_main = pd.read_csv('../data/df_processed.csv')

    # Linear regression to check correlation
    X = df_main.drop(['Success_all'], axis = 1)
    y = df_main['Success_all']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=420)

    feature_list = list(X.columns)

    # Grid Search Cross-validation
    parameters = {
        'n_estimators':(10, 30, 50), 
        'max_depth':(4,5,6,8,10,15),
        'min_samples_split': (2, 4, 8),
        'min_samples_leaf': (4,8,12,16)
    }

    rf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
    rf.fit(X_train, np.ravel(y_train))

    print('The accuracy on training set is: ', rf.best_score_)
    print('The best parameters are: ', rf.best_params_)

    rf_train = RandomForestClassifier(max_depth = 15,
                                      min_samples_leaf = 4, 
                                      min_samples_split = 8,
                                      n_estimators = 50)
    rf_train.fit(X_train, np.ravel(y_train))
    y_pred_train = rf_train.predict(X_train)

    # Importance of features on training set
    importances = rf_train.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10,10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_list)
    plt.xlabel('Relative Importance')
    plt.savefig('../reports/figures/rf_importance_plot.png')