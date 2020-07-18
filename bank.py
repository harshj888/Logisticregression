#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libraies
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


# loading bank data 
data = pd.read_csv("bank-full.csv",sep=';')


# In[3]:


data.head(10)


# In[4]:


data.shape    


# In[5]:


#find categorical variables
categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)


# In[6]:


# view the categorical variables
data[categorical].head()


# In[7]:


# check missing values in categorical variables
data[categorical].isnull().sum()


# In[8]:


# view frequency of categorical variables
for var in categorical: 
    print(data[var].value_counts())


# In[9]:


# check for cardinality in categorical variables
for var in categorical:
    print(var, ' contains ', len(data[var].unique()), ' labels')


# In[10]:


# find numerical variables
numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[11]:


data[numerical].head()


# In[12]:


# check missing values in numerical variables
data[numerical].isnull().sum()


# In[13]:


# check missing values in numerical variables
data[numerical].isnull().sum()


# In[14]:


# view summary statistics in numerical variables
print(round(data[numerical].describe()),2)


# In[15]:


data1 = data[data['y'] == 'yes']
data2 = data[data['y'] == 'no']


# In[16]:


fig, ax = plt.subplots(2, 2, figsize=(12,10))

b1 = ax[0, 0].bar(data1['loan'].unique(),height = data1['loan'].value_counts(),color='#000000')
b2 = ax[0, 0].bar(data2['loan'].unique(),height = data2['loan'].value_counts(),bottom = data1['loan'].value_counts(),color = '#DC4405')
ax[0, 0].title.set_text('Loan')
#ax[0, 0].legend((b1[0], b2[0]), ('Yes', 'No'))
ax[0, 1].bar(data1['month'].unique(),height = data1['month'].value_counts(),color='#000000')
ax[0, 1].bar(data2['month'].unique(),height = data2['month'].value_counts(),bottom = data1['month'].value_counts(),color = '#DC4405') 
ax[0, 1].title.set_text('Month')
ax[1, 0].bar(data1['job'].unique(),height = data1['job'].value_counts(),color='#000000')
ax[1, 0].bar(data1['job'].unique(),height = data2['job'].value_counts()[data1['job'].value_counts().index],bottom = data1['job'].value_counts(),color = '#DC4405') 
ax[1, 0].title.set_text('Type of Job')
ax[1, 0].tick_params(axis='x',rotation=90)
ax[1, 1].bar(data1['education'].unique(),height = data1['education'].value_counts(),color='#000000') #row=0, col=1
ax[1, 1].bar(data1['education'].unique(),height = data2['education'].value_counts()[data1['education'].value_counts().index],bottom = data1['education'].value_counts(),color = '#DC4405') 
ax[1, 1].title.set_text('Education')
ax[1, 1].tick_params(axis='x',rotation=90)
#ax[0, 1].xticks(rotation=90)
plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")
plt.show()


# In[17]:


fig, ax = plt.subplots(2, 2, figsize=(15,10))

b1 = ax[0, 0].bar(data1['marital'].unique(),height = data1['marital'].value_counts(),color='#000000')
b2 = ax[0, 0].bar(data1['marital'].unique(),height = data2['marital'].value_counts()[data1['marital'].value_counts().index],bottom = data1['marital'].value_counts(),color = '#DC4405') 
ax[0, 0].title.set_text('Marital Status')
#ax[0, 0].legend((b1[0], b2[0]), ('Yes', 'No'))
ax[0, 1].bar(data1['housing'].unique(),height = data1['housing'].value_counts(),color='#000000')
ax[0, 1].bar(data1['housing'].unique(),height = data2['housing'].value_counts()[data1['housing'].value_counts().index],bottom = data1['housing'].value_counts(),color = '#DC4405') 
ax[0, 1].title.set_text('Has housing loan')
ax[1, 0].bar(data1['contact'].unique(),height = data1['contact'].value_counts(),color='#000000')
ax[1, 0].bar(data1['contact'].unique(),height = data2['contact'].value_counts()[data1['contact'].value_counts().index],bottom = data1['contact'].value_counts(),color = '#DC4405') 
ax[1, 0].title.set_text('Type of Contact')
ax[1, 1].bar(data1['poutcome'].unique(),height = data1['poutcome'].value_counts(),color='#000000')
ax[1, 1].bar(data1['poutcome'].unique(),height = data2['poutcome'].value_counts()[data1['poutcome'].value_counts().index],bottom = data1['poutcome'].value_counts(),color = '#DC4405') 
ax[1, 1].title.set_text('Outcome of the previous marketing campaign')
plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")
plt.show()


# In[18]:


fig, ax = plt.subplots(2, 2, figsize=(12,10))

ax[0, 0].hist(data2['age'],color = '#DC4405',alpha=0.7,bins=20, edgecolor='white') 
ax[0, 0].hist(data1['age'],color='#000000',alpha=0.5,bins=20, edgecolor='white')
ax[0, 0].title.set_text('Age')
ax[0, 1].hist(data2['duration'],color = '#DC4405',alpha=0.7, edgecolor='white') 
ax[0, 1].hist(data1['duration'],color='#000000',alpha=0.5, edgecolor='white')
ax[0, 1].title.set_text('Contact duration')
ax[1, 0].hist(data2['campaign'],color = '#DC4405',alpha=0.7, edgecolor='white') 
ax[1, 0].hist(data1['campaign'],color='#000000',alpha=0.5, edgecolor='white')
ax[1, 0].title.set_text('Number of contacts performed')
ax[1, 1].hist(data2[data2['pdays'] != 999]['pdays'],color = '#DC4405',alpha=0.7, edgecolor='white') 
ax[1, 1].hist(data1[data1['pdays'] != 999]['pdays'],color='#000000',alpha=0.5, edgecolor='white')
ax[1, 1].title.set_text('Previous contact days')
plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")
plt.show()


# In[19]:


bank=pd.get_dummies(data,drop_first=True)
columns=bank.head()


# In[20]:


#Splitting the data into training and testing
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(bank,test_size=0.2)
train_data1=train_data.reset_index()
test_data1=test_data.reset_index()
train_data1=train_data1.drop("index",axis=1)
test_data1=test_data1.drop("index",axis=1)
train_data1
X1_train=train_data1.iloc[:,0:42]


# In[21]:


#Buliding the logistics model
import statsmodels.formula.api as sm
m1=sm.logit("y_yes~X1_train",data=train_data1).fit()
m1.summary()
m1.summary2()

train_pred=m1.predict(train_data1)
train_data1


# In[22]:


# probability value
train_data1["model_pred"]=np.zeros(36168)
train_data1.loc[train_pred>=0.50,"model_pred"]=1


# In[23]:


#Checking for model accuracy
from sklearn.metrics import classification_report
train_classification=classification_report(train_data1["y_yes"],train_data1["model_pred"])
train_classification


# In[24]:


#Confusion matrix
confusion_matrix_train=pd.crosstab(train_data1["y_yes"],train_data1["model_pred"])
confusion_matrix_train


# In[25]:


#Accuracy
train_accuracy=(31154+780)/36168
train_accuracy


# In[26]:


from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(train_data1["y_yes"],train_data1["model_pred"])
plt.plot(fpr,tpr);plt.xlabel("FALSE POSITIVE RATE");plt.ylabel("TRUE POSITIVE RATE")
roc_auc_train=metrics.auc(fpr,tpr)


# In[27]:


X2_test=test_data1.iloc[:,0:42]
import statsmodels.formula.api as sm
m2=sm.logit("y_yes~X2_test",data=test_data1).fit()
m2.summary()
m2.summary2()


# In[28]:


test_pred=m2.predict(test_data1)
test_data1


# In[29]:


#Setting the probability value
test_data1["model_pred0"]=np.zeros(9043)
test_data1.loc[test_pred>=0.50,"model_pred0"]=1


# In[30]:


#Checking for model accuracy
from sklearn.metrics import classification_report
test_classification=classification_report(test_data1["y_yes"],test_data1["model_pred0"])
test_classification


# In[31]:


#Confusion matrix
confusion_matrix_test=pd.crosstab(test_data1["y_yes"],test_data1["model_pred0"])
confusion_matrix_test


# In[32]:


#Accuracy
test_accuracy=(7796+386)/9043
test_accuracy


# In[33]:


#ROC Curve
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(test_data1["y_yes"],test_data1["model_pred0"])
plt.plot(fpr,tpr);plt.xlabel("FALSE POSITIVE RATE");plt.ylabel("TRUE POSITIVE RATE")
roc_auc_test=metrics.auc(fpr,tpr)


# In[34]:


# accuracy: (tp + tn) / (p + n)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_data1["y_yes"],test_data1["model_pred0"])
print('Accuracy: %f' % accuracy)


# In[35]:


# precision tp / (tp + fp)
from sklearn.metrics import precision_score
precision = precision_score(test_data1["y_yes"],test_data1["model_pred0"])
print('Precision: %f' % precision)


# In[36]:


# recall: tp / (tp + fn)
from sklearn.metrics import recall_score
recall = recall_score(test_data1["y_yes"],test_data1["model_pred0"])
print('Recall: %f' % recall)


# In[37]:


# f1: 2 tp / (2 tp + fp + fn)
from sklearn.metrics import f1_score
f1 = f1_score(test_data1["y_yes"],test_data1["model_pred0"])
print('F1 score: %f' % f1)


# In[38]:


# ROC AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_data1["y_yes"],test_data1["model_pred0"])
print('ROC AUC: %f' % auc)


# In[39]:


# confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_data1["y_yes"],test_data1["model_pred0"])
print(matrix)

