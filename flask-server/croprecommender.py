from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

PATH = '/Users/nc23629-keerthana/Desktop/hack/flask-server/Data-processed/crop_recommendation.csv'
df = pd.read_csv(PATH)

df['label'].unique()

df.dtypes

df['label'].value_counts()

# sns.heatmap(df.corr(),annot=True)

"""### Seperating features and target label"""

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

# """# Decision Tree"""

# from sklearn.tree import DecisionTreeClassifier

# DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

# DecisionTree.fit(Xtrain,Ytrain)

# predicted_values = DecisionTree.predict(Xtest)
# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('Decision Tree')
# print("DecisionTrees's Accuracy is: ", x*100)

# print(classification_report(Ytest,predicted_values))

from sklearn.model_selection import cross_val_score

# # Cross validation score (Decision Tree)
# score = cross_val_score(DecisionTree, features, target,cv=5)

# score

# """### Saving trained Decision Tree model"""

# import pickle
# # Dump the trained Naive Bayes classifier with Pickle
# DT_pkl_filename = 'DecisionTree.pkl'
# # Open the file to save as pkl file
# DT_Model_pkl = open(DT_pkl_filename, 'wb')
# pickle.dump(DecisionTree, DT_Model_pkl)
# # Close the pickle instances
# DT_Model_pkl.close()

# """# Guassian Naive Bayes"""

# from sklearn.naive_bayes import GaussianNB

# NaiveBayes = GaussianNB()

# NaiveBayes.fit(Xtrain,Ytrain)

# predicted_values = NaiveBayes.predict(Xtest)
# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('Naive Bayes')
# print("Naive Bayes's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# Cross validation score (NaiveBayes)
# score = cross_val_score(NaiveBayes,features,target,cv=5)
# score

# """### Saving trained Guassian Naive Bayes model"""

# import pickle
# # Dump the trained Naive Bayes classifier with Pickle
# NB_pkl_filename = 'NBClassifier.pkl'
# # Open the file to save as pkl file
# NB_Model_pkl = open(NB_pkl_filename, 'wb')
# pickle.dump(NaiveBayes, NB_Model_pkl)
# # Close the pickle instances
# NB_Model_pkl.close()

# """# Support Vector Machine (SVM)"""

# from sklearn.svm import SVC

# SVM = SVC(gamma='auto')

# SVM.fit(Xtrain,Ytrain)

# predicted_values = SVM.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('SVM')
# print("SVM's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# # Cross validation score (SVM)
# score = cross_val_score(SVM,features,target,cv=5)
# score

# """# Logistic Regression"""

# from sklearn.linear_model import LogisticRegression

# LogReg = LogisticRegression(random_state=2)

# LogReg.fit(Xtrain,Ytrain)

# predicted_values = LogReg.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('Logistic Regression')
# print("Logistic Regression's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# # Cross validation score (Logistic Regression)
# score = cross_val_score(LogReg,features,target,cv=5)
# score

# """### Saving trained Logistic Regression model"""

# import pickle
# # Dump the trained Naive Bayes classifier with Pickle
# LR_pkl_filename = 'LogisticRegression.pkl'
# # Open the file to save as pkl file
# LR_Model_pkl = open(DT_pkl_filename, 'wb')
# pickle.dump(LogReg, LR_Model_pkl)
# # Close the pickle instances
# LR_Model_pkl.close()

"""# Random Forest"""

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score

"""### Saving trained Random Forest model"""

import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


print(classification_report(Ytest,predicted_values))




"""## Making a prediction"""
def model_prediction(data):
    
    prediction = RF.predict(data)
    print(prediction)
    return prediction
    