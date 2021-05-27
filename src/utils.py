import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import pandas as pd

def plot_corr(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('../reports/figures/corr_plot.png')

def processSubset(feature_list, X_train, y_train, y):
    model = sm.OLS(y_train,X_train[list(feature_list)])
    regr = model.fit()
    RSS = ((regr.predict(X_train[list(feature_list)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def forward(predictors, X_train, y_train, y):
    remaining_predictors = [p for p in X_train.columns if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p], X_train, y_train, y))
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    return best_model

def report_best_scores(results, n_top = 3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")