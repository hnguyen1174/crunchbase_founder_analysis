import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import *

if __name__ == "__main__":

    df_main = pd.read_csv('../data/df_processed.csv')

    # Linear regression to check correlation
    X = df_main.drop(['Success_all'], axis = 1)
    y = df_main['Success_all']

    OLS_model=sm.OLS(y,X)
    result=OLS_model.fit()
    print(result.summary2())

    feature_list = list(X.columns)

    plot_corr(df_main.drop(['Success_all'], axis = 1))