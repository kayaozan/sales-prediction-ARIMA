WORK IN PROGRESS

# Sales Prediction via ARIMA Model

This repository contains sample code that demonstrates how to build an ARIMA model to train and predict a time series data.

## Introduction
### Goal


### Data Exploration

It's a time series data of sales with weekly intervals. Each date has multiple entries for category, subcategory and season of products.

| Date | Category | SubCategory | Sales |
| --- | --- | --- | ---: |
| 2022-06-19 | Category1 | SubCategory1 | 15177 |
| 2022-06-19 | Category1 | SubCategory2 | 1079 |
| 2022-06-19 | Category2 | SubCategory1 | 224 |
| 2022-06-19 | Category2 | SubCategory2 | 86 |

It's always a good idea to start with a general picture, so I plotted the total sales of each date.

![sales-date_whole](https://github.com/user-attachments/assets/034f1619-dd17-44db-9b28-95a7ff0b3e38)


## Train & Test
Within `arima-train.py`, seasonal parameters are listed. Using `auto-arima`, the best parameters are selected and tested.

## Forecast
`arima-predict.py` uses the best model that is trained in the previous section.

Each prediction is done step wise and being used as input for the next prediction.
