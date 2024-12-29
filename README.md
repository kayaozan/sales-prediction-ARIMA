# Sales Prediction via ARIMA Model

This repository contains sample code that demonstrates how to build an ARIMA model to forecast future sales.

Since sales are observed to have a seasonal pattern, ARIMA is selected as the forecasting model.


## Train & Test
Within `arima-test.py`, seasonal parameters are listed. Using `auto-arima`, the best parameters are picked and tested.

## Forecast
`arima-predict.py` uses the best model that is trained in the previous section to forecast.
