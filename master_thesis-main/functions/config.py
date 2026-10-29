#This file has the imports, paths, and some constants

#IMPORTS
#--------------------------------------------------------------
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import seaborn as sns
import re
import glob
import textwrap
import json
import optuna
import umap
import optuna
import math
import ast
import itertools
from itertools import product
from collections import Counter
from sentence_transformers import SentenceTransformer
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import timedelta
from exif import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, cross_validate, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway
from scipy.stats import skew

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

#PATHS
#--------------------------------------------------------------------

#Data
#---------------------------------------------------------------------
cgm_path = r'C:\\Users\\aoezek\\Desktop\\Thesis\\data\\Continuous_glucose_monitoring_final_dataset.xlsx'
metadata_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\data\Participants_baseline_characteristics.xlsx'
#LLM (gemini flash 2.5 API) meal image annotation results
meal_data_path = r"C:\Users\aoezek\Desktop\Thesis\gemini_results_final\meal_labels_output_adjusted.xlsx"
#---------------------------------------------------------------------

#processed data files / intermediate results (which were used later in other tasks)
#---------------------------------------------------------------------

#clean metadata file already has the age groups (binary) and bins, created in metadata_visualisation.ipynb
metadata_clean_path = r"C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\metadata_clean.xlsx" 

#created and saved in fasting_times.ipynb
fasting_times_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\average_fasting_times.xlsx'

#created and saved in metadata_visualisation.ipynb
age_gender_map_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\age_gender_map.xlsx'

#created and saved in clustering.ipynb
cluster_map_path =r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\cluster_assignments_ppgr_only.xlsx'

#main dataset with meal events and their matched 2 hour CGM windows
merged_cgm_meal_path =r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\analysis_ready_data.xlsx'

#created in cgm_data_feature_extraction.ipynb
cgm_extracted_features_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\cgm_extracted_features_46.xlsx'

#created in ppgr_analysis, has the iAUC etc features
models_data_ppgr_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\models_data_ppgr.xlsx'

#iAUC prediction ready data, filtered is the one excluding the 2 participants with <10 meals. these come from ppgr_analysis_data_prep.ipynb
iauc_prediction_data_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\tuned_engineered_task1_models_ready_df.xlsx'
iauc_prediction_data_filtered_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\tuned_engineered_task1_models_ready_df_NOT_FILTERED.xlsx'

#created in the geminioutput_decode.ipynb file
merged_meal_meta_valid_path=r"C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\merged_meal_meta_valid_participants.xlsx"
meal_path=r'C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\meal_labels_output_clean_decoded.xlsx'

#prepared in time_series_data_preparation.ipynb
time_series_data_path = r"C:\Users\aoezek\Desktop\master_thesis\master_thesis\finalised_clean_notebooks\excels\task2_timeseries_data.xlsx"

#-------------------------------------------------------------------
#constants
#-------------------------------------------------------------------
drop_cols = [
    "meal_id", "Participant_id", "datetime", "date",
    "glucose_norm", "time_since_meal_min",
    "peak_increment", "time_to_peak", "estimated_total_weight_g",
    "time_hour", "meal_time_rad",
]
meal_cols = [
    "total_calories", "carbs_g", "protein_g", "fat_g", "fiber_g",
    "glycemic_index", "ultra_processed",
    "food_processing_level", "added_sugar", "fluid_or_solid", "diet_type",
    "sin_time", "cos_time"
]
glucose_plus_time = [    "mean", "std", "min", "max",
    "lt_3.9", "3.9-10", "gt_10",
    "Mean_Speed", "SD_Speed",
    "Mean_Acceleration", "SD_Acceleration",
    "carbs_x_glucoseSD", "Baseline_Glucose",     
    "carbs_x_baseline", "sin_time", "cos_time"

]
meta_cols = [
    "sex", "age", "body_weight", "body_height", "BMI",
    "Systolic_BP", "Diastolic_BP", "Resting_heart_rate"
]
glucose_cols = [
    "mean", "std", "min", "max",
    "lt_3.9", "3.9-10", "gt_10",
    "Mean_Speed", "SD_Speed",
    "Mean_Acceleration", "SD_Acceleration",
    "carbs_x_glucoseSD", "Baseline_Glucose",     
    "carbs_x_baseline"
]
circadian_cols = ["mesor", "amplitude", "acrophase", "period",
                  "carbs_x_acrophase", 'meal_dist_to_circadian', ]

glucose_plus_time = [
    "mean", "std", "min", "max",
    "lt_3.9", "3.9-10", "gt_10",
    "Mean_Speed", "SD_Speed",
    "Mean_Acceleration", "SD_Acceleration",
    "carbs_x_glucoseSD", "baseline_glucose",
    "carbs_x_baseline", "sin_time", "cos_time"
]

days = ["Tag 1", "Tag 2", "Tag 3", "Tag 4", "Tag 5", "Tag 6", "Tag 7", "Tag 8", "Tag 9", "Tag 10"] #11 and 12 are control days and should be excluded, study includes the first 10 days
string_cols=['sex','completed_10days', 'completed_12days', 'complete_cgm_data'] 

bins_binary=[18,59,77]
labels_binary=["19-32","59-76"]
