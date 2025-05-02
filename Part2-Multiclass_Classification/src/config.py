# basic imports
import os, random
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
import dill as pickle
import pandas_ta as ta
from pathlib import Path


# warnings
import warnings
warnings.filterwarnings('ignore')

# plotting & outputs
from pprint import pprint
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-bright')
pd.set_option("display.max_columns",None)
import seaborn as sns

# sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve, plot_roc_curve
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

# import classifiers
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier,
                              RandomForestClassifier, 
                              ExtraTreesClassifier,
                              GradientBoostingClassifier, 
                              BaggingClassifier,
                              VotingClassifier, 
                              StackingClassifier)

from xgboost import XGBClassifier 
from catboost import CatBoostClassifier
import lightgbm as lgb





