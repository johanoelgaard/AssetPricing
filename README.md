# Seminar: Asset Pricing and Financial Markets

This repository contains the code for reproducing the results of the paper titeled Forecasting Electricity Prices in Western Denmark:
A Weather-Based Approach written by Alon Clausen (smr136@alumni.ku.dk) and Johan Ølgaard (jlh601@alumni.ku.dk) at part of the seminar named Asset Pricing and Financial Markets at the University of Copenhagen.

The repository is structured as follows:
- The [`PreProcessing`](PreProcessing) folder contains 2 notebooks for processing the raw data and a .py file for the functions used in the notebooks
- The [`Modeling`](Modeling) folder contains 4 notebooks for modeling the data. One notebook for each model and a notebook for model comparison. The folder also contains a .py file for the functions used in the notebooks

To get from raw data to the final results, the notebooks should be run in the following order:
1. [`PreProcessing`](PreProcessing) folder
    - [`loadbulkweather.ipynb`](PreProcessing/loadbulkweather.ipynb)
    - [`combinedata.ipynb`](PreProcessing/combinedata.ipynb)

2. [`Modeling`](Modeling) folder
    - [`SARIMA.ipynb`](Modeling/ARIMA.ipynb), [`SARIMAX.ipynb`](Modeling/LSTM.ipynb), and [`LSTM.ipynb`](Modeling/LSTM.ipynb) can be run in any order
    - [`modelevaluation.ipynb`](Modeling/modelevaluation.ipynb)
