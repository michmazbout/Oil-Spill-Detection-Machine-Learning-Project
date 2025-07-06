import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import requests
from io import StringIO
import pooch
# url is the link for the Dataset on github
url = 'https://raw.githubusercontent.com/michmazbout/ML-Project/main/oil_spill.csv'
#location = '/content/gdrive/MyDrive/oil_spill.csv'
df  = pd.read_csv(url)
#We use .head() just to have an idea of the dataset
df.head()
# We are checking how many oil spills and how many non oil spills we have
df['target'].value_counts()
from sklearn.model_selection import train_test_split
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
#Splitting our dataset into 80% training and 20% test set
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(np.sum(y_test,axis =0))
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#Initializing all four classifiers and setting their parameters
rfc = RandomForestClassifier(n_estimators=3048,max_depth=2024,random_state=1)
gbc = GradientBoostingClassifier(n_estimators=2048,learning_rate=0.1)
dtc = DecisionTreeClassifier(max_depth=864,random_state=1)
#gnb = GaussianNB(var_smoothing=0.000000000245) for this smoothing we will catch the most oil spills out of all the rest of the of the functions but it will also catch 15 false positives
gnb = GaussianNB(var_smoothing=0.000000015) # best in my opinion 9 oil spils out of 12 and only 7 fp
#fitting our training dataset
rfc.fit(x_train,y_train)
gbc.fit(x_train,y_train)
dtc.fit(x_train,y_train)
gnb.fit(x_train,y_train)
#predicting using the classifier the test set
y_pred_rfc = rfc.predict(x_test)
y_pred_gbc = gbc.predict(x_test)
y_pred_dtc = dtc.predict(x_test)
y_pred_gnb = gnb.predict(x_test)
#Calculating the confusion matrix for each test result set
from sklearn.metrics import confusion_matrix
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
cm_gbc = confusion_matrix(y_test,y_pred_gbc)
cm_dtc = confusion_matrix(y_test,y_pred_dtc)
cm_gnb = confusion_matrix(y_test,y_pred_gnb)
# Calculating the accuracy scores for our test prediction results
acc_rfc = accuracy_score(y_test,y_pred_rfc)
acc_gbc = accuracy_score(y_test,y_pred_gbc)
acc_dtc = accuracy_score(y_test,y_pred_dtc)
acc_gnb = accuracy_score(y_test,y_pred_gnb)
#Printing the accuracy scores we calculated to use in report and compare.
print(f'\nThe accuracy of the model Random Forest Classifier is {acc_rfc:.1%}')
print(f'\nThe accuracy of the model Gradient Boosting Classifier is {acc_gbc:.1%}')
print(f'\nThe accuracy of the model Decision Tree Classifier is {acc_dtc:.1%}')
print(f'\nThe accuracy of the model Gaussian NB Classifier is {acc_gnb:.1%}')
#Note each of these plots below is made to be in its proper section
#in the google collab so if you run the code as is without splitting
#the sections then you will have all confusion plots on top of each other
#Plotting the confusion matrix for the random forest classifier
sns.heatmap(cm_rfc,cmap="Reds",annot=True,vmax=10,vmin=0)
#Plotting the confusion matrix for the gradiant boosting classifier
sns.heatmap(cm_gbc,cmap="Blues",annot=True,vmax=10,vmin=0)
#Plotting the confusion matrix for the decision tree classifier
sns.heatmap(cm_dtc,cmap="Greys",annot=True,vmax=15,vmin=0)
#Plotting the confusion matrix for the Gaussian naive bayes classifier
sns.heatmap(cm_gnb,cmap="viridis",annot=True,vmax=20,vmin=0)
import shap
#X_test_summary = shap.sample(x_test, 10)
# Kernal Explainer for the Ramdom Forest Classifier
explainer = shap.KernelExplainer(gbc.predict,x_test)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test)
# Kernal Explainer for the Gradient Boosting Classifier
explainer = shap.KernelExplainer(gbc.predict,x_test)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test)
# Kernal Explainer for the Decision Tree Classifier
explainer = shap.KernelExplainer(dtc.predict,x_test)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test)
# Kernal Explainer for the Guassian Naive Bayes Classifier
explainer = shap.KernelExplainer(gnb.predict,x_test)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test)
