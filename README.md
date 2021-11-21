<div align="center">

# Monthly Baseflow Dataset for the CONUS

provides an opportunity to analyze large-scale baseflow trends under global change 🔥<br>

</div>
<br>

<div align="center">

![](https://user-images.githubusercontent.com/29588684/142756866-7e22814d-2e78-4fad-8035-86eee529bb10.gif)

</div>
<br>

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Train and Evaluate](#train-and-evaluate)
    - [Baseflow Simulation](#baseflow-simulation)
- [License](#license)
<br>


<a name="introcution"></a>
## 📌&nbsp;&nbsp;Introduction
To fill the gaps in time-varying baseflow datasets, we introduced a machine learning approach called the long short-term memory (LSTM) network to develop a monthly baseflow dataset.

To better train across basins, we compared the standard LSTM with four variant architectures using additional static properties as input. Results show that three variant architectures (Joint, Front, and EA-LSTM) perform better than the standard LSTM, with median Kling-Gupta efficiency across basins greater than 0.85.

Based on Front LSTM, the monthly baseflow dataset with 0.25° spatial resolution across the contiguous United States from 1981 to 2020 was obtained, which can be downloaded from the [release page](https://github.com/xiejx5/BaseFlowCONUS/releases).
<br>
<br>

<a name="structure"></a>
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

<a name="usage"></a>
## ℹ️&nbsp;&nbsp;Usage

### Data Preparation
- First, download watershed.zip from the [release page](https://github.com/xiejx5/watershed_delineation/releases)
- Next, unzip and open watershed.exe, clip start to execute an example
<br>

### Hyperparameter Tuning
```bash
python run.py -m tuner=optuna
```
<br>

### Train and Evaluate
```bash
python run.py -m model=front dataset.eco=CPL, NAP, NPL
```

```bash
python run.py -m model=front datasplit=full dataset.eco=CPL, NAP, NPL
```
<br>

### Baseflow Simulation
```bash
from src import simulate

checkpoint = 'saved/train/front/CPL/models/model_latest.pth'
simulate(checkpoint)
```
<br>

<a name="license"></a>
## 🚀&nbsp;&nbsp;License
