import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    df_main = pd.read_csv('../data/df_processed.csv')

    # Linear regression to check correlation
    X = df_main.drop(['Success_all'], axis = 1)
    y = df_main['Success_all']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=420)

    models_fwd = pd.DataFrame(columns=["RSS", "model"])
    tic = time.time()
    predictors = []
    for i in range(1,len(X_train.columns)+1):    
        models_fwd.loc[i] = forward(predictors, X_train, y_train, y)
        predictors = models_fwd.loc[i]["model"].model.exog_names
    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")

    fig, ax = plt.subplots(figsize=(7, 7))
    models_fwd['RSS'].plot()
    plt.xlabel('Model Complexity')
    plt.ylabel('RSS')
    plt.savefig('../reports/figures/stepwise_plot.png')

    print("-----------------")
    print("Foward Selection:")
    print("-----------------")
    print(models_fwd.loc[10, "model"].params)

    # We choose the 10 features above for our baseline model
    chosen_features = list(models_fwd.loc[10, "model"].params.index)

    X_train_bestsubset = X_train[chosen_features]
    X_test_bestsubset = X_test[chosen_features]

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred_bestsubset = linreg.predict(X_test)
    y_pred_bestsubset = np.where(y_pred_bestsubset > 0.5, 1, 0)

    #Baseline MSE and Accuracy
    print('The baseline MSE is {:.2%}'.format(mean_squared_error(y_test, y_pred_bestsubset)))
    print('The baseline accuracy is {:.2%}'.format(sum(y_pred_bestsubset == y_test)/len(y_test)))