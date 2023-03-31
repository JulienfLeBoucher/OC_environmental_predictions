# Energy consumptions and GHG emissions predictions.

In this project, I use [environmental data collected in Seattle](https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy) in order to predict energy consumptions and GHG emissions of the non-residential properties.

Briefly, I discovered :
- many regressors, their pros and cons, and how to tune their hyperparameters thanks to grid-searches and cross-validation.
- the impact of pre-processing inputs and transforming targets. *(very important)*
- how to properly evaluate models' performances with cross-validation.
- model explainability with SHAP.

Here are links to my notebooks (displayed with nbviewer) :

- [Exploration notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_environmental_predictions/blob/main/exploration_notebook.ipynb)
- [Energy consumptions prediction notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_environmental_predictions/blob/main/total_energy_predictions.ipynb)
- [GHG emissions prediction notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_environmental_predictions/blob/main/GHG_emissions_prediction.ipynb)
