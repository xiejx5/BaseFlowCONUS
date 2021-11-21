<div align="center">

# Estimating gridded monthly baseflow for CONUS


</div>
<br><br>![Fig_Gif](https://user-images.githubusercontent.com/29588684/142756866-7e22814d-2e78-4fad-8035-86eee529bb10.gif)




**Contents**
- [Introduction](#introduction)
- [Main Ideas Of This Template](#main-ideas-of-this-template)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Guide](#guide)
    - [How To Learn?](#how-to-learn)
    - [Main Project Configuration](#main-project-configuration)
    - [Experiment Configuration](#experiment-configuration)
    - [Workflow](#workflow)
    - [Logs](#logs)
    - [Experiment Tracking](#experiment-tracking)
    - [Inference](#inference)
    - [Callbacks](#callbacks)
- [Best Practices](#best-practices)
    - [Miniconda](#miniconda)
    - [Automatic Code Formatting](#automatic-code-formatting)
    - [Environment Variables](#environment-variables)
    - [Data Version Control](#data-version-control)
    - [Installing Project As A Package](#support-installing-project-as-a-package)
    - [Tests](#tests)
- [Tricks](#tricks)
- [Other Repositories](#other-repositories)
<br>


## Introduction
We estimated the 0.25° monthly gridded baseflow across the contiguous US from 1980 to 2018 using a machine learning approach called the long short-term memory (LSTM) network.

## Project Structure
The directory structure of the project looks like this:
```
├── configs                 <- Hydra configuration files
│   ├── trainer                 <- Configurations of Lightning trainers
│   ├── datamodule              <- Configurations of Lightning datamodules
│   ├── model                   <- Configurations of Lightning models
│   ├── callbacks               <- Configurations of Lightning callbacks
│   ├── logger                  <- Configurations of Lightning loggers
│   ├── optimizer               <- Configurations of optimizers
│   ├── experiment              <- Configurations of experiments
│   │
│   ├── config.yaml             <- Main project configuration file
│   └── config_optuna.yaml      <- Configuration of Optuna hyperparameter search
│
├── data                    <- Project data
│
├── logs                    <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks               <- Jupyter notebooks
│
├── tests                   <- Tests of any kind
│
├── source
│   ├── architectures           <- PyTorch model architectures
│   ├── callbacks               <- PyTorch Lightning callbacks
│   ├── datamodules             <- PyTorch Lightning datamodules
│   ├── datasets                <- PyTorch datasets
│   ├── models                  <- PyTorch Lightning models
│   ├── transforms              <- Data transformations
│   ├── utils                   <- Utility scripts
│   │   ├── inference_example.py    <- Example of inference with trained model
│   │   └── template_utils.py       <- Some extra template utilities
│   │
│   └── train.py                <- Contains training pipeline
│
├── run.py                  <- Run training with chosen experiment configuration
│
├── .env                    <- File for storing environment variables
├── .gitignore              <- List of files/folders ignored by git
├── .pre-commit-config.yaml <- Configuration of hooks for automatic code formatting
├── conda_env_gpu.yaml      <- File for installing conda environment
├── requirements.txt        <- File for installing python dependencies
├── LICENSE
└── README.md
```
<br>

