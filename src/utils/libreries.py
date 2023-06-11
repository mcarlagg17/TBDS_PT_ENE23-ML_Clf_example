#####################################################################
# -------------------------------------------------------------------
# Variables to change depend on the dataset
# -------------------------------------------------------------------

model_of_ = 'c'         # c: classification models or 
                        # r: regression models (SUPERVISED MODELS)

encode = True           # True: if we need to encode or 
                        # False: if not 

unsupervised = False     # True: if is used unsupervised estimators 
                        # False: if not

supervised = True       # True: if is used supervised estimators 
                        # False: if not

deep_learning = False   # True: if is used deep learning 
                        # False: if not

imbalanced_data = True  # True: if is used imbalanced data 
                        # False: if not

# ==================================================================
####################################################################

# Generic libreries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import scipy.stats as ss
import time
import os 
import statsmodels.api as sm
import matplotlib.colors as mcolors
#from phik import phik_matrix
import pickle
from sklearn.metrics import make_scorer
from scipy.stats import norm, skew


# APIs
#import requests
import json

# Data cleaning/ data analisys
if encode:
    from sklearn.preprocessing import LabelEncoder 
    from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


# Machine Learning models
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, RepeatedKFold, learning_curve)

from sklearn.pipeline import Pipeline

if supervised:

    if model_of_ == 'c':
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
        from sklearn.ensemble import (RandomForestClassifier,BaggingClassifier,VotingClassifier,
                                    AdaBoostClassifier,GradientBoostingClassifier,)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        
    elif model_of_ == 'r':
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import ElasticNet
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB,BernoulliNB
        from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
        from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
        from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
        from sklearn.svm import SVR
        from xgboost import XGBRegressor
        
    else:
        print('Model type error')

if unsupervised:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import dendrogram, linkage

if deep_learning :
    from sklearn.neural_network import MLPClassifier    

# Prepare and train models
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

# Model metrics evaluation
if model_of_=='c':
    from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                                recall_score, f1_score, roc_auc_score, classification_report,jaccard_score,roc_curve, auc)
elif model_of_=='r':
    from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
else: 
    print('Model type error')



# MCGG functions (created by me)
from utils.functions import *
