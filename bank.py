import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Importing the dataset
bank=pd.read_excel("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 3- Logistic Regression\\Bank_correct.xlsx")
# EDA 
plt.hist(bank.age) # Not a normal distribution but is a required variable
plt.boxplot(bank.age) # has a large number of outliers

plt.hist(bank.job)# Needs to be converted to dummy variables
jobs=pd.get_dummies(bank["job"],drop_first=True)
jobs.head()

plt.hist(bank.marital)
mstatus=pd.get_dummies(bank["marital"],drop_first=True)
mstatus.head()

plt.hist(bank.education)
ed=pd.get_dummies(bank["education"],drop_first=True)
ed.head()

plt.hist(bank.default)
defaulty=pd.get_dummies(bank["default"],drop_first=True)
defaulty.head() # One reason for the imbalanced set is they have given majority data of defaulters

plt.hist(bank.balance) # required variable

plt.hist(bank.housing)
hl=pd.get_dummies(bank["housing"],drop_first=True)
hl.head()

plt.hist(bank.loan)
L=pd.get_dummies(bank["loan"],drop_first=True)
L.head()

plt.hist(bank.contact)# Not required can be deleted

plt.hist(bank.day)# Not required can be deleted
plt.hist(bank.month)# Not required can be deleted

plt.hist(bank.duration) # required variable

# Deleteing all columns which are not required or either converted to dummy variables
bank.drop(["job","marital","education","default","housing","loan","contact","day","month","campaign","pdays","previous","poutcome"],axis=1,inplace=True)

# Adding all dummy variables to the bank dataframe
bank=pd.concat([bank,jobs,mstatus,ed,defaulty,hl,L],axis=1)
bank.info()

# Apart from the Y variable all variables are continous

# Checking if there are any null variables

bank.isnull().sum() # There is no null variable

# Checking for imbalanced data set
plt.hist(bank.y) # Clearly an imbalanced data set

final=bank
final.info()

# Building a model without considering the imbalance
x=final.drop("y",axis=1)
y=final["y"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

predictions=logmodel.predict(x_test)

# To calculate the accuracy of model

# Method 1- Classification Report

from sklearn.metrics import classification_report
classification_report(y_test,predictions)
# Confusion matric

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions) # Model is claiming 89% accuracy

# Solving the issue of imbalanced data set

minority_class_len=len(final[y=="yes"])
print(minority_class_len) # 5289

majority_class_indices=final[y=="no"].index
print(majority_class_indices)


random_majority_indices=np.random.choice(majority_class_indices,minority_class_len)

print(len(random_majority_indices))#5289

minority_class_indices=final[y=="yes"].index
print(len(minority_class_indices))#5289

under_sample_indices=np.concatenate([random_majority_indices,minority_class_indices])
under_sample=final.loc[under_sample_indices]
print(under_sample)

# Preparing the logistic regression model after eliminating the imbalanced data set

x=under_sample.drop("y",axis=1)
y=under_sample["y"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression().fit(x_train,y_train)

predictions=logmodel.predict(x_test)

# Checking the accuracy of the model
# Method 1- Classification Report

from sklearn.metrics import classification_report
classification_report(y_test,predictions)
# Confusion matric

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions) # 77% accuracy


