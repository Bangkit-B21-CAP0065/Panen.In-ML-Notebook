# Panen.In-ML-Notebook
Notebook and python code for Machine Learning Crop Prediction Program

# Overview

Crop folder contains python file to run crop prediction model. Additional file including 4 Tensorflow models and architectures (.h5 format) and 2 .csv files as input.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all python dependencies.

You can use requirements.txt

```bash
pip install -r requirements.txt
```

## Library version used

Python 3.8.7

Pandas 1.2.4

Numpy 1.19.5

Tensorflow 2.5.0

## Run

Run from your terminal with command line arguments:
```python
python3 crop.py <subrounds> <city> <type of crop>
```
Example:
```python
python3 crop.py 2 bogor jagung
```

## Other Files

.ipynb files can be opened and run using Jupyter or Google Colab (https://colab.research.google.com/)
