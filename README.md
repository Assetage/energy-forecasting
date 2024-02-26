Hourly Energy Consumption
==============================
This project based on https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption 
### 📖 Description
### **PJM Hourly Energy Consumption Data**
PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.

The hourly power consumption data comes from PJM's website and are in megawatts (MW).

The regions have changed over the years so data may only appear for certain dates per region.
   
A short description of the project.
------------
The primary objective of this project is to establish a well-organized project structure and effectively implement diverse techniques.   
Project structure is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.   
For config handling  <a target="_blank" href="https://hydra.cc/docs/intro/">Hydra library</a>  is used .   
Notebooks folder contain notebook done based on the work of <a target="_blank" href="https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost">PT2: Time Series Forecasting with XGBoost</a> (ToDo: add more features).   

To be done:
1. Add more features
2. Try other models
3. Make better analysis notebook
4. Cover more functions with tests

## **Project Organization**
------------
```plaintext
├── LICENSE
├── README.md
├── data
│   └── raw                                     <- The original, immutable data dump.
├── models                                      <- Trained and serialized models, model predictions, or model summaries
├── notebooks                                   <- Jupyter notebooks
├── outputs                                     <- Hydra logs
├── pyproject.toml                              <- .toml file to define a project package and make it installable (pip install -e .)
├── requirements.txt                            <- requirements for correct installation of the project package
└── src                                         <- Source code for use in this project
     └── energy_forecasting                      <- Sub-folder of the source code for possible additional applications to the package
         ├── __init__.py                         <- Makes src a Python module
         ├── __main__.py                         <- Main function which allows to call entire train / optimize / predict pipelines
         ├── conf
         │   ├── config.yaml
         │   ├── feature_params                  <- Configs for features
         │   ├── optimizer_params                <- Configs for optimizer
         │   ├── path_config                     <- Configs for paths   
         │   ├── pipeline                        <- Configs for pipeline
         │   ├── splitting_params                <- Configs for splitting params
         │   └── train_params                    <- Configs for train
         ├── data
         │   ├── __init__.py
         │   └── make_dataset.py                 <- Scripts for time series splitting of the dataset
         ├── entities                            <- Scripts for creating dataclasses
         ├── features                            <- Scripts to turn raw data into features for modeling
         ├── model_params_optimizer.py           <- Optimizer pipeline code
         ├── models                              <- Scripts to train models and making predictions based on them
         │   ├── __init__.py
         │   ├── predict_model.py                
         │   └── train_model.py
         ├── predict_pipeline.py                 <- Predict pipeline code
         ├── train_pipeline.py                   <- Train pipeline code
         └── utils
             ├── __init__.py
             └── utils.py                        <- Additional utility scripts
├── tests                                        <- Project tests

```
--------
## ⛏**Package installation**
The package can be installed by cloning the repo and then by typing: 
```shell
pip install .
```
The package will be available in your pip under `energy-forecasting` name.

--------
## ⚡**Run training**
Once the package is installed, one can initiate a training pipeline. For training the "pipeline" argument value is "train". Take into account that "train_params" argument allows a user to train a specific model. For the current release of the package only "LinearRegression", "RandomForestRegressor" and "MLPRegressor" models are available with "lr", "rf" and "nn" shortcuts, respectively.

Run model train for Random Forest Regressor:
```shell  
energy_forecasting pipeline=train train_params=rf
```

--------
## ⚡**Run prediction**
Once the package is installed, one can initiate a training pipeline. For prediction the "pipeline" argument value is "predict". In order to use a specific model use "predict_params" argument values of "lr", "rf" or "nn".

Run prediction with a Random Forest Regressor:  
```shell
energy_forecasting pipeline=predict predict_params=rf
```

--------
## ⚡**Run optimization**
Once the package is installed, one can initiate a training pipeline. For model optimization the "pipeline" argument value is "optimize". In order to use a specific model to optimize use "optimize_params" argument values of "lr", "rf" or "nn".

Run model optimization for a Random Forest Regressor:  
```shell
energy_forecasting pipeline=predict optimize_params=rf
```
--------
## **Additional information**
To ensure that you follow the development workflow, please setup the pre-commit hooks:

```shell
pre-commit install
```

You can run the tests with the following commands:

```shell
# Unit tests
make test
```