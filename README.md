# Seminar: Asset Pricing and Financial Markets
University of Copenhagen, Fall 2024

This repository contains the code for reproducing the results of the paper [`Post-2021 Forecasting of Electricity Prices in Western
Denmark: A Weather-Based Approach`](Seminar_Asset_Prices_and_Financial_Markets.pdf) written by Alon Clausen (smr136@alumni.ku.dk) and Johan Ølgaard (jlh601@alumni.ku.dk) as part of the seminar named Asset Pricing and Financial Markets at the University of Copenhagen.

The repository is structured as follows:
- The [`PreProcessing`](PreProcessing) folder contains 2 notebooks for processing the raw data and a .py file for the functions used in the notebooks
- The [`Modelling`](Modelling) folder contains 4 notebooks for modeling the data. One notebook for each model and a notebook for model comparison. The folder also contains a .py file containing the library of custom functions used in the notebooks

To get from raw data to the final results, the notebooks should be run in the following order:
1. [`PreProcessing`](PreProcessing) folder
    - [`loadbulkweather.ipynb`](PreProcessing/loadbulkweather.ipynb)
    - [`combinedata.ipynb`](PreProcessing/combinedata.ipynb)

2. [`Modelling`](Modelling) folder
    - [`SARIMA.ipynb`](Modelling/SARIMA.ipynb), [`SARIMAX.ipynb`](Modelling/SARIMAX.ipynb), and [`LSTMmodel.ipynb`](Modelling/LSTMmodel.ipynb) can be run in any order
    - [`ModelEvaluation.ipynb`](Modelling/ModelEvaluation.ipynb)
