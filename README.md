# Data-analysis-and-classification-on-food-using-different-models
Description: 
Using the dataset of the nutrient, I perform a basic in depth analysis in regards to the different data that was in the dataset. 
Dealing with such a large set of columns, i used columns that are most impactful through the use of correlation. 
- Another way of getting the columns is by using PCA

I wrote my reasons as to why i use certain technique and observation throughout the entire project in the pdf file

NOTE: From Random-Forest onwards, there are no explanation as they are extra models that i have done on the side. 

Skills set used: 
- Basic Cleaning of dataset
- Splitting and merging of datasets 
- Correlations of the databases between different columns (for solid and liquid )
- Display : Confusion matrix Display
- KNN-Classifier (cross validation) 
- Logistic Regression
- Neural Network : Relu , sigmoid
- Random Forest
- Bagging and out of bag error
- Dropout
- Boosting 

Library used: 
- !pip install mlxtend
- !pip install seaborn
- !pip install tensorflow
- import seaborn as sns
- !pip install openpyxl
- import pandas as pd
- import matplotlib.pyplot as plt
- import numpy as np
- from sklearn.model_selection import train_test_split
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.metrics import accuracy_score
- import matplotlib.pyplot as plt
- from sklearn.preprocessing import MinMaxScaler
- from sklearn.model_selection import cross_val_score
- from sklearn.linear_model import LogisticRegression
- import tensorflow as tf
- import numpy as np
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import MinMaxScaler
- from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
- from tensorflow.keras.models import Sequential
- from tensorflow.keras.layers import Dense
- from sklearn.metrics import classification_report
- from sklearn.metrics import f1_score
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.ensemble import BaggingClassifier
- from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
- from sklearn.ensemble import GradientBoostingClassifier

The data was obtained from Food Standards Australia and New Zealand: https://www.foodstandards.gov.au/science/monitoringnutrients/afcd/Pages/default.aspx



