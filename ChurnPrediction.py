
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#Load libraries
from fancyimpute import KNN   
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[3]:


#Loading the Data
train = pd.read_csv('../input/Train_data.csv', error_bad_lines=False)
test = pd.read_csv('../input/Test_data.csv', error_bad_lines=False)


# In[4]:


train.head(10)


# In[5]:


#Create dataframe with missing percentage
missing_val_train = pd.DataFrame(train.isnull().sum())
missing_val_test = pd.DataFrame(test.isnull().sum())


# In[6]:


missing_val_train


# In[7]:


missing_val_test


# In[8]:


df = train.copy()


# In[9]:


# #Plot boxplot to visualize Outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(train['total day minutes'])


# In[10]:


cnames = ["account length","area code","number vmail messages","total day minutes","total day calls","total day charge","total eve minutes","total eve calls","total eve charge","total night minutes","total night calls","total night charge","total intl minutes","total intl calls","total intl charge","number customer service calls"]


# In[ 11]:


#Detect and replace with NA
# #Extract quartiles
for i in cnames:
    q75, q25 = np.percentile(train.iloc[:,i], [75 ,25])

# #Calculate IQR
     iqr = q75 - q25

# #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)

# #Replace with NA
   train[train.iloc[[:,i] < minimum] = np.nan
    train[train.loc[:,i] > maximum] = np.nan

# #Calculate missing value
    missing_val = pd.DataFrame(marketing_train.isnull().sum())

# #Impute with KNN
    train = pd.DataFrame(KNN(k = 3).complete(train), columns = train.columns)


# In[13]:


##Correlation analysis
#Correlation plot
df_corr = train.loc[:,cnames]


# In[14]:


df_corr


# In[16]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[19]:


#Chisquare test of independence
#Save categorical variables
cat_names = ["state","phone number","international plan","voice mail plan"]


# In[20]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train['Churn'], train[i]))
    print(p)


# In[27]:


train.head(10)


# In[54]:


train = train.drop(["phone number","total day minutes","total eve minutes","total night minutes","total intl minutes"], axis=1)


# In[52]:


test = test.drop(["phone number","total day minutes","total eve minutes","total night minutes","total intl minutes"], axis=1)


# In[55]:


test.shape


# In[30]:


df = train.copy()


# In[56]:


train.shape


# In[31]:


#Normality check
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(train['area code'], bins='auto')


# In[32]:


cnames = ["account length","area code","number vmail messages","total day calls","total day charge","total eve calls","total eve charge","total night calls","total night charge","total intl calls","total intl charge","number customer service calls"]


# In[33]:


#Nomalisation
for i in cnames:
    print(i)
    train[i] = (train[i] - min(train[i]))/(max(train[i]) - min(train[i]))


# In[34]:


train.head()


# In[57]:


#Nomalisation
for i in cnames:
    print(i)
    test[i] = (test[i] - min(test[i]))/(max(test[i]) - min(test[i]))


# In[36]:


#Import Libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[79]:


#replace target categories with 0 or 1
test['Churn'] = test['Churn'].replace('False.', 0)
test['Churn'] = test['Churn'].replace('True.', 1)


# In[80]:


test['Churn']


# In[81]:


#Divide data into train and test
X_train = train.iloc[:, 0:15]
Y_train = train.iloc[:,15]

X_test = test.iloc[:, 0:15]
Y_test = test.iloc[:,15]


# In[82]:


Y_train.head(5)


# In[83]:


one_hot_encoded_X_train = pd.get_dummies(X_train)
one_hot_encoded_X_test = pd.get_dummies(X_test)
final_train, final_test = one_hot_encoded_X_train.align(one_hot_encoded_X_test,
                                                                    join='left', 
                                                                    axis=1)


# In[85]:


#Decision Tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(final_train, Y_train)

#predict new test cases
C50_Predictions = C50_model.predict(final_test)

#Create dot file to visualise tree  #http://webgraphviz.com/
# dotfile = open("pt.dot", 'w')
# df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names = marketing_train.columns)


# In[90]:


#build confusion matrix
#from sklearn.metrics import confusion_matrix 
#CM = confusion_matrix(Y_test, Y_pred)
CM = pd.crosstab(Y_test, C50_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
(FN*100)/(FN+TP)

#Results
#Accuracy: 85.12
#FNR: 30.80


# In[91]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(final_train, Y_train)


# In[92]:


RF_Predictions = RF_model.predict(final_test)


# In[97]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(Y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 100%
#FNR: 100%


# In[112]:


#Let us prepare data for logistic regression
#replace target categories with Yes or No
train['Churn'] = train['Churn'].replace(0, 'False')
train['Churn'] = train['Churn'].replace(1, 'True')


# In[113]:


train['Churn']


# In[107]:


#Create logistic data. Save target variable first
train_logit = pd.DataFrame(train['Churn'])


# In[108]:


#Add continous variables
train_logit = train_logit.join(train[cnames])


# In[109]:


train_logit.head()


# In[114]:


##Create dummies for categorical variables
cat_names = ["state","international plan","voice mail plan"]

for i in cat_names:
    temp = pd.get_dummies(train[i], prefix = i)
    train_logit = train_logit.join(temp)


# In[115]:


#Create logistic data. Save target variable first
test_logit = pd.DataFrame(test['Churn'])


# In[116]:


#Add continous variables
test_logit = test_logit.join(test[cnames])


# In[117]:


##Create dummies for categorical variables
cat_names = ["state","international plan","voice mail plan"]

for i in cat_names:
    temp = pd.get_dummies(test[i], prefix = i)
    test_logit = test_logit.join(temp)


# In[118]:


train =train_logit.copy()
test = test_logit.copy()


# In[119]:


#select column indexes for independent variables
train_cols = train.columns[1:30]


# In[121]:


#Built Logistic Regression
import statsmodels.api as sm
from pandas.core import datetools

logit = sm.Logit(train['Churn'], train[train_cols]).fit()

logit.summary()
#Accuracy = 87%
#FNR = 71.65%

# In[125]:


#KNN implementation
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(final_train, Y_train)


# In[126]:


#predict test cases
KNN_Predictions = KNN_model.predict(final_test)


# In[130]:


#build confusion matrix
CM = pd.crosstab(Y_test, KNN_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 86.92
#FNR: 91.46


# In[131]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(final_train, Y_train)


# In[132]:


#predict test cases
NB_Predictions = NB_model.predict(final_test)


# In[134]:


#Build confusion matrix
CM = pd.crosstab(Y_test, NB_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 59.20
#FNR: 41.07

