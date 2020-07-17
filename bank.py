#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


# view summary statistics in numerical variables
print(round(data[numerical].describe()),2)


# In[14]:


data1 = data[data['y'] == 'yes']
data2 = data[data['y'] == 'no']


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


# Declare feature vector and target variable
X = data.drop(['y'], axis=1)
y = data['y']


# In[19]:


data.head(1)


# In[20]:


data.drop(['pdays','poutcome','campaign','previous'] ,axis=1, inplace=True)


# In[21]:


data = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan'])


# In[22]:


sns.heatmap(data.corr())
plt.show()


# In[ ]:





# In[ ]:





# In[23]:


X = data.iloc[:,1:]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[24]:


# check the shape of X_train and X_test
X_train.shape, X_test.shape


# In[25]:


# check data types in X_train

X_train.dtypes


# In[26]:


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[27]:


# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[28]:


# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[29]:


# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()


# In[30]:


X_train[numerical].describe()


# In[31]:


# Removing outliers and cap maximum values
max_bal = data['balance'].quantile(0.90)
print('Maximum range for balance = ',max_bal)

max_duration = data['duration'].quantile(0.90)
print('Maximum range for duration = ',max_duration)


# In[32]:


def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['balance'] = max_value(df3, 'balance', max_bal)
    df3['duration'] = max_value(df3, 'duration', max_duration)


# In[33]:


X_train.balance.max(), X_test.balance.max()


# In[34]:


X_train.duration.max(), X_test.duration.max()


# In[35]:


# Enncode Categorical Values
X_train.head()


# In[36]:


#Encode Categorical Values 
import category_encoders as ce
encoder_owner = ce.BinaryEncoder(cols=['y'])
X_train = encoder_owner.fit_transform(X_train)
X_test = encoder_owner.transform(X_test)

encoder_owner = ce.BinaryEncoder(cols=['education'])
X_train = encoder_owner.fit_transform(X_train)
X_test = encoder_owner.transform(X_test)

encoder_owner = ce.BinaryEncoder(cols=['contact'])
X_train = encoder_owner.fit_transform(X_train)
X_test = encoder_owner.transform(X_test)

encoder_owner = ce.BinaryEncoder(cols=['month'])
X_train = encoder_owner.fit_transform(X_train)
X_test = encoder_owner.transform(X_test)


# In[37]:


X_train.head()


# In[38]:


X_test.head()


# In[39]:


X_train.describe()


# In[40]:


cols = X_train.columns


# In[41]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[42]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[43]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[44]:


X_train.describe()


# In[45]:


X_test.describe()


# In[46]:


X_test.head(5)


# In[47]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)


# In[48]:


# Predicted Results
y_pred_test = logreg.predict(X_test)

y_pred_test


# In[49]:


y_pred_test = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)


# In[50]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:





# In[ ]:




