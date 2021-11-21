<div align="center">

# Monthly Baseflow Dataset for the CONUS

</div>
<br>

<div align="center">

![](https://user-images.githubusercontent.com/29588684/142756866-7e22814d-2e78-4fad-8035-86eee529bb10.gif)

</div>
<br>

- [Introduction](#introduction)
- [Folder Structure](#project-structure)
- [Usage](#introduction)
    - [Data Preparation](#data-preparation)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Train and Evaluate](#train-and-evaluate)
    - [Baseflow Simulation](#baseflow-simulation)
- [Citation](#citation)
<br>


## 📌&nbsp;&nbsp;Introduction
To fill the gaps in time-varying baseflow datasets, we introduced a machine learning approach called the long short-term memory (LSTM) network to develop a monthly baseflow dataset.

To better train across basins, we compared the standard LSTM with four variant architectures using additional static properties as input. Results show that three variant architectures (Joint, Front, and EA-LSTM) perform better than the standard LSTM, with median Kling-Gupta efficiency across basins greater than 0.85.

Based on Front LSTM, the monthly baseflow dataset with 0.25° spatial resolution across the contiguous United States from 1981 to 2020 was obtained, which can be downloade from the [release page](https://github.com/xiejx5/BaseFlowCONUS/releases).
<br>
<br>

## ⚡&nbsp;&nbsp;Project Structure
```yaml
baseflow/
├── configs                 <- Hydra configuration files
│   ├── constant                <- Folder paths and constants
│   ├── dataset                 <- Configs of Pytorch dataset
│   ├── datasplit               <- Split dataset into train and test
│   ├── hydra                   <- Configs of Hydra logging and launcher
│   ├── loss                    <- Configs of loss function
│   ├── model                   <- Configs of Pytorch model architectures
│   ├── optimizer               <- Configs of optimizer
│   ├── trainer                 <- Configs of validation metrics and trainer
│   ├── tuner                   <- Configs of Optuna hyperparameter search
│   └── config.yaml             <- Main project configuration file
│
├── data                    <- Baseflow, time series, and static properties
│
├── logs                    <- Logs generated by Hydra and PyTorch loggers
│
├── saved                   <- Saved evaluation results and model parameters
│
├── src
│   ├── datasets                <- PyTorch datasets
│   ├── datasplits              <- Dataset splitter for train and test
│   ├── models                  <- PyTorch model architectures
│   ├── trainer                 <- Class managing training process
│   ├── utils                   <- Utility scripts for metric logging
│   ├── evaluate.py             <- Model evaluation piplines
│   ├── perpare.py              <- Data preparation piplines
│   └── simulate.py             <- Simulate gridded baseflow
│
├── run.py                  <- Run pipeline with chosen configuration
│
├── main.py                 <- Main process for the whole project
│
├── .gitignore              <- List of files/folders ignored by git
├── requirements.txt        <- File for installing python dependencies
├── LICENSE
└── README.md
```
<br>

## ℹ️&nbsp;&nbsp;Usage

### Data Preparation
<br>

### Hyperparameter Tuning
<br>

### Train and Evaluate
<br>

### Baseflow Simulation
<br>

## 🚀&nbsp;&nbsp;Citation

