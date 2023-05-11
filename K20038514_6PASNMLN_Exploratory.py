#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:43:39 2023

@author: k20038514
"""

# Step 0: Import libraries

import numpy as np                # Arrays and mathematical function
import pandas as pd               # Dataframes
import matplotlib.pyplot as plt   # Plotting- Remove if not necessary
import seaborn as sns             # For the confusion matrix
from numpy import mean            # Mean
from numpy import std             # Standard Deviation
from matplotlib import pyplot     # For plotting
from sklearn.metrics import confusion_matrix # To compute the confusion matrices
from sklearn.metrics import roc_curve, auc # Load in the ROC libraries
from sklearn.metrics import accuracy_score # Importing accuracy score

# Step 1: Loading and inspecting data

# Load in training data and convert categoricals to numeric:
vals = pd.read_csv('training_validation.csv') # Demographics and values 
vals_name = vals.columns # Getting dataframe column names
vals.head() # To check that columns names are correct
vals.info() # Get descriptive statistics
vals.isnull().sum() # Check to see if there are any null values... there are none

# To get size of vals dataframe:
def get_dataframe_size(vals):
    return vals.shape

vals_size = get_dataframe_size(vals)
print(vals_size) # Tuple (652, 296)

# Convert diagnosis column from categorical to numerical
vals['Diagnosis'] = pd.factorize(vals['Diagnosis'])[0] # AD = 1, control = 0

# Work out mean age, number of AD and control participants
mean_age = vals['Age'].mean() # Calculate mean age
print('The mean age is', mean_age) # Report (mean age = 74)

AD_num = vals['Diagnosis'].value_counts()[1] # Calculate how many participants are assigned '1', aka have Alzheimer's
Control_num = vals['Diagnosis'].value_counts()[0] # Calculate how many participants are assigned '0', aka are healthy controls

print('There are', AD_num, 'Alzheimers participants and', Control_num, 'controls in this dataset') # Report this

# Splitting data into Diagnosis and Grey matter data
vals_name = vals.columns

X = vals[vals_name[4:]] # Columns with grey matter data
Y = vals[vals_name[2]] # Columns with diagnosis data

# Step 2: Split data into training and testing

from sklearn.model_selection import StratifiedShuffleSplit

# The dataset is split 80% for training & validation, and 20% for testing
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=3)

# The enumerate method allows us to print training+validation and testing indices across all folds i 
# (NOTE that in this case, there is only one fold)
for i, (trainval_ind, test_ind) in enumerate(sss.split(X, Y)):
    print(f"Fold {i}:")
    print(f"  Train+Val: index={trainval_ind}")
    print(f"  Test:  index={test_ind}")
    
# Apply the split to the data
# Create separate training+validation arrays
GM_train = X.iloc[trainval_ind,:] # GM
AD_train = Y.iloc[trainval_ind] # AD
 # iloc for pandas dataframe

# Create testing arrays
GM_test = X.iloc[test_ind,:] 
AD_test = Y.iloc[test_ind]

# To check whether data is split or balanced:
print(f'In the training and validation set, there are {np.sum(AD_train==0)} healthy controls, and {np.sum(AD_train==1)} patients with AD.')
# 312 controls, 209 with AD

# Step 3: Standardisation of data

# Getting mean, standard deviation (SD) and Z-scored values for X 
mean_GM_train = np.mean(GM_train)                # Mean
std_GM_train = np.std(GM_train)                  # SD
X_z = (GM_train - mean_GM_train)/std_GM_train     # Z-score

# Getting mean, standard deviation (SD) for Y, data does not need to be standardised as it only composes of 0s and 1s
mean_AD_train = np.mean(AD_train)                # Mean
std_AD_train = np.std(AD_train)                  # SD

# Step 4: Logistic regression
from sklearn.linear_model import LogisticRegression # To run a logistic regression 

# 4.1 Fit model on training data
logreg = LogisticRegression(max_iter=1000) # C = 1 automatically set
logreg.fit(GM_train,AD_train)

# 4.2 Predict target variable on test data
LR_pred = logreg.predict(GM_test)

# 4.3 Create an output dataframe which contains the predicted labels
LR_Predicted = pd.DataFrame({'Predicted': LR_pred}) # Dataframe
LR_Predicted.to_csv('LR_predicted_labels.csv', index=False) # Convert dataframe to csv file

# 4.4 Calculate precision accuracy
LR_accuracy = accuracy_score(AD_test, LR_pred) 
print(LR_accuracy) # Accuracy 0.862...

# 4.5 Using grid search to determine best paramters
from sklearn.model_selection import GridSearchCV # Grid search 

param_grid = {'C': [0.1, 1, 10, 100]} # Set the possibilities of C
grid = GridSearchCV(LogisticRegression(), param_grid, cv= 10)
grid.fit(GM_train, AD_train) # Fit these to the model

print("Best parameter: ", grid.best_params_) # Best value of C is 1, so don't need to update the model

# 4.6 Compute a confusion matrix
plt.rcParams.update({'font.size': 14}) # Increase font size
cm = confusion_matrix(GM_test, LR_pred) # Actual and predicted values
sns.heatmap(cm,annot=True) # If True, write the data value in each cell
plt.xlabel('Predicted label') # Add X label
plt.ylabel('True label') # Add Y label
plt.title('Logistic Regression Confusion Matrix') # Add title

# 4.7 Produce ROC curve
pos_pred_prob = logreg.predict_proba(GM_test)[:,1] # Calculate the predicted probability of the positive class
fpr, tpr, thresholds = roc_curve(AD_test, pos_pred_prob) # Run the ROC curve function
roc_auc = auc(fpr, tpr) # Calculate the area under the curve (AUC)
print(roc_auc) # 0.895...

# Plotting the true positive rate against the false positive rate at the different classification thresholds
plt.figure(figsize=(8, 6)) # Defining size of figure
plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc) # This is the ROC curve
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Baseline
plt.xlim([0.0, 1.0]) # X limits
plt.ylim([0.0, 1.05]) # Y limits
plt.xlabel('False Positive Rate = 1 - Specificity') # X label
plt.ylabel('True Positive Rate = Sensitivity') # Y label
plt.title('Receiver operating characteristic (ROC) curve') # Title
plt.legend(loc="lower right") # Key to show what red line is and area of curve
plt.show() # Show plot


# Step 5: K-Nearest Neighbours
from sklearn.preprocessing import StandardScaler    # Scaling the data
from sklearn.neighbors import KNeighborsClassifier # KNN classifier
from sklearn.pipeline import make_pipeline # For pipeline

# Step 5.1 Define number of neighbours and fit classifier 
n_neighbours = 3
knn = KNeighborsClassifier(n_neighbours)
sc = StandardScaler()
GM_train = sc.fit_transform(GM_train) # Define data scaler and apply to train and test
GM_test = sc.transform(GM_test)

# 5.2 Making the classifier 
classifier = make_pipeline(sc, knn) # Combining sc and knn into a single model for classification
classifier.fit(GM_train, AD_train) # Fitting pipeline to training data

# 5.3 Generate predicted labels and store in a csv file
KNN_pred = classifier.predict(GM_test) # Generate class predictions on test data

KNN_Predicted = pd.DataFrame({'Predicted': KNN_pred}) # Dataframe
KNN_Predicted.to_csv('KNN_predicted_labels.csv', index=False) # Convert dataframe to csv file

# 5.4 Compute and display ROC curves and confusion matrix for data
fpr, tpr, thresholds = roc_curve(AD_test, KNN_pred) # Obtain false positive rate (fpr), true positive rate (tpr) and thresholds at different cutoffs
roc_auc = auc(fpr, tpr) # Define area under the ROC curve

plt.rcParams.update({'font.size': 14}) # Increase font size
cm = confusion_matrix(AD_test, KNN_pred) # Compute the confusion matrix 
sns.heatmap(cm,annot=True) # If True, write the data value in each cell
plt.xlabel('Predicted label') # Add X label
plt.ylabel('True label') # Add Y label
plt.title('KNN Confusion Matrix') # Add title

KNN_accuracy = accuracy_score(AD_test,KNN_pred) # ...and accuracy
print('The automatically calculated accuracy is', KNN_accuracy) # 0.755...

# Plotting the true positive rate against the false positive rate at the different classification thresholds
plt.figure(figsize=(8, 6)) # Defining size of figure
plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc) # This is the ROC curve
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Baseline
plt.xlim([0.0, 1.0]) # X limits
plt.ylim([0.0, 1.05]) # Y limits
plt.xlabel('False Positive Rate = 1 - Specificity') # X label
plt.ylabel('True Positive Rate = Sensitivity') # Y label
plt.title('Receiver operating characteristic (ROC) curve') # Title
plt.legend(loc="lower right") # Key to show what red line is and area of curve
plt.show() # Show plot

# Step 6: Setting up the Extra Trees Classifier

# Step 6.1 Initialising ExtraTrees, training data and making predictions
from sklearn.ensemble import ExtraTreesClassifier # Importing the classifier

ET_classifier = ExtraTreesClassifier() # Initializing the ExtraTrees model
ET_classifier.fit(GM_train, AD_train) # Training the data
classifier_pred = ET_classifier.predict(GM_test) # Making predictions

# Step 6.2: Visualising results
cm = confusion_matrix(AD_test, classifier_pred) # Actual and predicted values
sns.heatmap(cm,annot=True) # If True, write the data value in each cell

# Step 6.3 Obtaining accuracy score

accuracy_score(AD_test,classifier_pred) # Accuracy is 0.809

# Step 6.4:  Adjusting hyperparamters to obtain higher accuracy 
from sklearn.model_selection import cross_val_score # Cross validation score
from sklearn.model_selection import RepeatedStratifiedKFold # Stratified K-Fold
from sklearn.ensemble import ExtraTreesClassifier # The ExtraTrees classifier

# 6.5 Re-define train datasets to avoid complication
ET_GM = GM_train 
ET_AD = AD_train

model = ExtraTreesClassifier() # Define model

# 6.6 Evaluate the model
cv = RepeatedStratifiedKFold(n_splits=11, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, ET_GM, ET_AD, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise') # Report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores))) # Accuracy is 0.874

# 6.7 Adjusting hyperparamaters
# An important hyperparameter of extra trees is the number of decision trees used in the ensemble
# This code attempts to vary the number of trees to analyse the effect this has on the accuracy of the model

# 6.8 Get the dataset
def get_dataset():
	ET_GM, ET_AD 
	return ET_GM, ET_AD

# 6.9 Get a list of models to evaluate
def get_models():
 models = dict()

# 6.10 Define how many trees to be considered
 n_trees = [10, 50, 100, 500, 1000]
 for n in n_trees:
     models[str(n)] = ExtraTreesClassifier(n_estimators=n)
 return models
 
# 6.11 Evaluate model with K-Fold cross-validation
def evaluate_model(model, ET_X, ET_Y):

 cv = RepeatedStratifiedKFold(n_splits=11, n_repeats=3, random_state=1) # Define procedure for evaluation

# 6.12 Collect results
 tree_scores = cross_val_score(model, ET_X, ET_Y, scoring='accuracy', cv=cv, n_jobs=-1)
 return tree_scores
 
# 6.13 Get the models to evaluate
models = get_models()

# 6.14 Evaluate and store results
results, names = list(), list()
for name, model in models.items():
 tree_scores = evaluate_model(model, ET_GM, ET_AD)  # Evaluate model
 results.append(tree_scores) # Store results
 names.append(name)
 print('>%s %.3f (%.3f)' % (name, mean(tree_scores), std(tree_scores))) # Summarise performance, each time
pyplot.boxplot(results, labels=names, showmeans=True) # Plot model performances to compare
pyplot.show() # 500 trees is the most effective approach

# 6.15: Running predictions with new improved model
ET_classifier_best = ExtraTreesClassifier(n_estimators=500) # Run best classifer with optimum estimators 
ET_classifier_best.fit(ET_GM, ET_AD)

best_pred = ET_classifier_best.predict(GM_test) # Predictions with 500 trees

ET_predicted = pd.DataFrame({'predictions': best_pred}) # Store predicted labels in dataframe
ET_predicted.to_csv('predictions.csv', index=False) # Write the dataframe to a CSV file

ET_accuracy = accuracy_score(AD_test, best_pred)
print("The accuracy score for the improved model is:", ET_accuracy)

# 6.16 Compute and display ROC curves and confusion matrix for data
fpr, tpr, thresholds = roc_curve(AD_test, ET_predicted) # Obtain false positive rate (fpr), true positive rate (tpr) and thresholds at different cutoffs
roc_auc = auc(fpr, tpr) # Define area under the ROC curve

# Plotting the true positive rate against the false positive rate at the different classification thresholds
plt.figure(figsize=(8, 6)) # Defining size of figure
plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc) # This is the ROC curve
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Baseline
plt.xlim([0.0, 1.0]) # X limits
plt.ylim([0.0, 1.05]) # Y limits
plt.xlabel('False Positive Rate = 1 - Specificity') # X label
plt.ylabel('True Positive Rate = Sensitivity') # Y label
plt.title('Receiver operating characteristic (ROC) curve') # Title
plt.legend(loc="lower right") # Key to show what red line is and area of curve
plt.show() # Show plot

# 6.17 Compute a confusion matrix 
plt.rcParams.update({'font.size': 14}) # Increase font size
cm = confusion_matrix(AD_test, best_pred) # Actual and predicted values
# If True, write the data value in each cell
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted label') # Add X label
plt.ylabel('True label') # Add Y label
plt.title('ExtraTrees Confusion Matrix') # Add title

# Step 8: Decide on the best model
print("The best model which most accurately predicts labels is Logistic Regression, which a prediction accuracy of",LR_accuracy) # Best value of C is 1, so don't need to update the model

