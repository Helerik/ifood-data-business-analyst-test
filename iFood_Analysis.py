# !/usr/bin/env Python3
# Author: Erik Davino Vincent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SkLearn
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

# Hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.fmin import generate_trials_to_calculate

# Utilities
import winsound
import warnings


def main():

    # Import dataset:
    data = pd.read_csv("ml_project1_data.csv")
