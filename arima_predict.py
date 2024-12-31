import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

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

with open('arima_models', mode='rb') as f:
    models = pickle.load(f)

# A timeseries dataset that contains sales numbers by date, product category, subcategory etc.
data = pd.read_pickle('SalesRetailSeason')

# Determining the columns to select the subsection of dataset.
filterCols = data.select_dtypes(include=['object']).columns.tolist()
filterKeys = get_filters(data, filterCols)

# Number of predictions to forecast.
n_predict = 13

all_predictions = []
for filterKey in filterKeys:
    print(filterKey)

    mask = np.logical_and.reduce(
        [data[key] == value for key, value in filterKey.items()])
    data_filtered = data.copy()[mask].set_index('Date')[['Sales']]

    # Choosing the model with the corresponding filter keys
    params = [model for model in models if 
              all([model[key] == filterKey[key] for key in filterKey.keys()])][0]['params']

    y = [v.item() for v in data_filtered.values]
    prediction_values = []
    prediction_idx = pd.date_range(data_filtered.index.max(),
                                   periods=n_predict+1,
                                   freq='W',
                                   inclusive='right')

    for t in range(n_predict):
        model = ARIMA(y,
                      order=params['order'],
                      seasonal_order=params['seasonal_order'])
        model_fit = model.fit()
        output = model_fit.forecast()
        y_pred = int(output[0])
        prediction_values.append(y_pred)
        y.append(y_pred)

    predictions = pd.DataFrame(prediction_values,
                               index=prediction_idx,
                               columns=data_filtered.columns)
    all_predictions.append([filterKey,predictions])

    # Adding data and forecast seperately
    plt.plot(data_filtered, label='Actual Sales')
    plt.plot(predictions, label='Forecasted Sales', color='red')
    # Arranging labels etc.
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=30)
    plt.title('-'.join([v for v in filterKey.values()]))
    plt.show()
