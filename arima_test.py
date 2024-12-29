import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')

def get_filters(df : pd.DataFrame, cols: list) -> list[dict]:
    # Groups the DataFrame by its given columns,
    # returns the possible combinations of values in those columns
    
    filterKeys = []
    for row in df.groupby(cols).size().reset_index().drop(0,axis=1).iterrows():
        filterKeys.append({})
        for i, r in enumerate(row[1]):
            filterKeys[-1][cols[i]] = r

    return filterKeys

train_percentage = 0.8
season_params = [{'D':None,'m':1},
                 {'D':0,'m':52},
                 {'D':1,'m':52}]

# A timeseries dataset that contains sales numbers by day, product category, subcategory etc.
data = pd.read_pickle('SalesRetailSeason')

# Determining the columns to select the subsection of dataset.
filterCols = data.select_dtypes(include=['object']).columns.tolist()
filterKeys = get_filters(data, filterCols)

models = []
for filterKey in filterKeys:
    print('Filtering the dataset with:', filterKey)

    mask = np.logical_and.reduce(
        [data[key] == value for key, value in filterKey.items()])
    data_filtered = data.copy()[mask].set_index('Date')[['Quantity']]

    fits = []
    # Going through every seasonal parameters to find the best ones.
    for params in season_params:
        fits.append(auto_arima(data_filtered,
                               D=params['D'],
                               m=params['m'],
                               trace=True,
                               information_criterion='aicc',
                               suppress_warnings=True))
        
    # Comparing and choosing the best fit.
    bestscore = np.inf
    bestfit = None
    for fit in fits:
        if fit.aicc() < bestscore:
            bestscore = fit.aicc()
            bestfit = fit

    models.append(filterKey.copy())
    models[-1]['params'] = {'order': bestfit.order,
                            'seasonal_order':bestfit.seasonal_order}

    # Since this is a timeseries, train and test are split continuously
    X = data_filtered.values
    size = int(len(X) * train_percentage)
    train, test = X[0:size], X[size:len(X)]
    history = [x.item() for x in train]
    predictions = list()

    for t in range(len(test)):
        # Model is set up and fit with the bestfit arguments.
        model = ARIMA(history, order=bestfit.order, seasonal_order=bestfit.seasonal_order)
        model_fit = model.fit()
        # Forecast is generated with the model.
        output = model_fit.forecast()
        y_pred = output[0]
        predictions.append(y_pred)
        y_true = test[t]
        # Predicted value is added to the end of the train set.
        # That way, the model can be tested when predicted values are used as input
        # and the next value is predicted.
        history.append(y_pred)
        print('predicted=%f, expected=%f' % (y_pred, y_true))

    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.title('-'.join([v for v in filterKey.values()]))
    plt.show()

with open('arima_models', mode='wb') as f:
    pickle.dump(models,f)