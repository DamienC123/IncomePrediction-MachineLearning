# -*- coding: utf-8 -*-
"""
@author: Damien Chao

"""
import pandas as pd
#import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import accuracy_score

# Insert Directory
#os.chdir(r'C:\')
dataset = pd.read_csv("income.csv")

# Check shape of date, length, variable types 
print("Dataset length:", len(dataset))
print("Data shape initial:", dataset.shape)
dataset.dtypes
#dataset.head()

# Check for missing values
dataset.isna().sum()

# Group by occupation on income for a quick check of the significance
df = dataset.groupby('occupation')['income'].describe()
print(df)
# Group by occupation on income for a quick check of the significance

df = dataset.groupby('workclass')['income'].describe()
print(df)

# Checking significance of relationship and marital status
#df = dataset.groupby(['relationship','marital-status'])['income'].describe()
#print(df)


# Delete missing values since there is no way to average/handle the missing 
# workclass and occupations
dataset = dataset.dropna()
print("Dataset length after removing missing values:", len(dataset))


# Check if there are any duplicated rows in the dataset
dataset.duplicated().any()
dataset = dataset.drop_duplicates()
print("Dataset length after removing duplicates:", len(dataset))

# Check value counts for the categorical variables
#print(dataset.education.value_counts(), "\n")

# Handle categorical variables for education
dataset['education'] = dataset['education'].replace({
    'Preschool': 1,
    '1st-4th': 2,
    '5th-6th': 3,
    '7th-8th': 4,
    '9th': 5,
    '10th': 6,
    '11th': 7,
    '12th': 8,
    'HS-grad' : 9,
    'Some-college': 10,
    'Assoc-voc': 11,
    'Assoc-acdm': 12,
    'Bachelors': 13,
    'Masters': 14,
    'Prof-school': 15,
    'Doctorate': 16,
    })

#dataset['education'].head()

# Handle categorical variables for sex
dataset['sex'] = dataset['sex'].replace({
    'Male' : 0,
    'Female': 1
    })

#dataset['sex'].head()


# Handle the other categorical (workclass, marital-status, occupation, relationship,
# race) with dummy coding

dataset = pd.get_dummies(dataset, columns = ['workclass','marital-status',
                                             'occupation', 'relationship',
                                             'race'])

print("Data shape after preprocessing:", dataset.shape)

# Define the input and target variables
array = dataset.values
X = array[:,1:40]
y = array[:,0]


# Splitting the dataset into training and testing use
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.1,
                                                   random_state =123)
# Apply normalization on both train and testing dataset

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing data
X_test_norm = norm.transform(X_test)


#-----------------------------------------------------------------------------

# Part 2

# Logistic Regression Model
model1 = LogisticRegression(max_iter=2000)
model1.fit(X_train_norm, y_train)
test_score1 = model1.score(X_test_norm, y_test)
print("Testing Accuracy of LogR:", test_score1)

# Support Vector Machine for classification
model2 = SVC(max_iter=20000)
model2.fit(X_train_norm, y_train)
test_score2 = model2.score(X_test_norm, y_test)
print("Testing Accuracy of SVM:", test_score2)

# 10-fold crossvalidation with shuffle and random state of 123
kfold = KFold(n_splits=10, shuffle=True, random_state=123)

# Train and evaluate Log R model using 10 fold CV
model1 = LogisticRegression(max_iter=2000)
results1 = cross_val_score(model1, X_train_norm, y_train, cv=kfold)
print("Average Accuracy Score of LogR (Train):", results1.mean())


# Train and evaluate SVM model using 10 fold CV
model2 = SVC(max_iter=20000)
results2 = cross_val_score(model2, X_train_norm, y_train, cv=kfold)
print("Average Accuracy of SVM (Train):", results2.mean())


# Fine tuning - Optimize the Logistic Regression Model using training dataset
grid_params_lr1 = {
    'penalty': ['l1', 'l2'],
    'C': [1,10],
    'solver': ['liblinear']
}

# Best Parameter
grid_params_lr2 = {
    'penalty': [None, 'l2'],
    'C': [1,10],
    'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']
}

grid_params_lr3 = {
    'penalty': ['elasticnet'],
    'C': [1,10],
    'solver': ['saga'],
    'l1_ratio': [0,0.5,1]
}

gs_lr_result = GridSearchCV(model1, grid_params_lr2, cv=kfold).fit(X_train_norm, y_train)
print("Best score of LogR:", gs_lr_result.best_score_)

# Evaluate the trained Logistic Regression model using testing data
test_accuracy = gs_lr_result.best_estimator_.score(X_test_norm, y_test)
print("Accuracy in testing (LR):", test_accuracy)

# Check the paramater setting for the best selected model
gs_lr_result.best_params_


# Fine tuning - Optimize the trained SVM model using testing dataset
grid_params_svc = {
    'kernel': ['rbf'], # replace each case with different kernels (linear, poly, sigmoid)
    'C': [1,10], # tried to use 100 as well
    'degree': [3], # tried to use 8 as well
    'gamma': ['scale'], # tried to use auto as well
    'max_iter': [20000],
}

#svc = SVC(max_iter=20000)
gs_svc_result = GridSearchCV(model2, grid_params_svc, cv=kfold).fit(X_train_norm, y_train)
print("Best Score of SVM:", gs_svc_result.best_score_)

# Evaluate the trained SVM model using testing data

test_accuracy = gs_svc_result.best_estimator_.score(X_test_norm, y_test)
print("Accuracy in testing (SVM):", test_accuracy)

# Check the parameter settings for the best selected model
gs_svc_result.best_params_




#-----------------------------------------------------------------------------

# Part 3: Clustering

# Build a kmeans model and apply clustering (using 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=123).fit(X_train_norm)

# Check how many data samples in each cluster
unique_labels, unique_counts = np.unique(kmeans.labels_, return_counts=True)
dict(zip(unique_labels, unique_counts))

# Extract a prototype from each cluster 
kmeans_cluster_centers = kmeans.cluster_centers_
closest = pairwise_distances_argmin(kmeans_cluster_centers, X_train_norm)

# Show the two data samples that can represent the two clusters
dataset.iloc[closest, : ].T #Chose to transpose as it makes it more readable


# Check accuracy of the clustering model (training set)
y = array[:,0]

kmeans_labels = kmeans.labels_
accuracy = accuracy_score(y_train, kmeans_labels)
print("K Means Prediction Accuracy:", accuracy)

# Evaluate clustering accuracy with the testing set
kmeans_test_labels = kmeans.predict(X_test)
accuracy_test = accuracy_score(y_test, kmeans_test_labels)
print("K Means Testing Accuracy:", accuracy_test)
