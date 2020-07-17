#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading creditcard data 
df = pd.read_csv("creditcard.csv")


# In[3]:


df.shape


# In[4]:


df.head(5)


# In[5]:


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[6]:


df.head(5)


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)


# In[11]:


# view the categorical variables
df[categorical].head()


# ### card is the target variable.

# In[12]:


# check missing values in categorical variables
df[categorical].isnull().sum()


# In[13]:


# view frequency of categorical variables
for var in categorical: 
    print(df[var].value_counts())


# In[14]:


# check for cardinality in categorical variables
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')


# In[15]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[16]:


# check labels in owner variable
df['owner'].unique()


# In[17]:


pd.get_dummies(df.owner, drop_first=True, dummy_na=True).head()


# In[18]:


# check labels in selfemp variable
df['selfemp'].unique()


# In[19]:


pd.get_dummies(df.selfemp, drop_first=True, dummy_na=True).head()


# In[20]:


# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[21]:


df[numerical].head()


# In[22]:


# check missing values in numerical variables
df[numerical].isnull().sum()


# In[23]:


# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)


# In[24]:


# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='expenditure')
fig.set_title('')
fig.set_ylabel('expenditure')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='months')
fig.set_title('')
fig.set_ylabel('months')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='age')
fig.set_title('')
fig.set_ylabel('age')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='active')
fig.set_title('')
fig.set_ylabel('active')


# In[25]:


# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.expenditure.hist(bins=10)
fig.set_xlabel('expenditure')
fig.set_ylabel('card')


plt.subplot(2, 2, 2)
fig = df.months.hist(bins=10)
fig.set_xlabel('months')
fig.set_ylabel('card')


plt.subplot(2, 2, 3)
fig = df.age.hist(bins=10)
fig.set_xlabel('age')
fig.set_ylabel('card')


plt.subplot(2, 2, 4)
fig = df.active.hist(bins=10)
fig.set_xlabel('active')
fig.set_ylabel('card')


# In[26]:


#find IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[27]:


# Declare feature vector and target variable
X = df.drop(['card'], axis=1)
y = df['card']


# In[28]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[29]:


# check the shape of X_train and X_test
X_train.shape, X_test.shape


# In[30]:


# check data types in X_train

X_train.dtypes


# In[31]:


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[32]:


# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[33]:


# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[34]:


# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()


# In[35]:


# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()


# In[36]:


# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()


# In[37]:


X_train[numerical].describe()


# In[38]:


# Removing outliers and cap maximum values
max_expn = df['expenditure'].quantile(0.90)
print('Maximum range for exoendture = ',max_expn)

max_months = df['months'].quantile(0.90)
print('Maximum range for months = ',max_months)

max_age = df['age'].quantile(0.90)
print('Maximum range for age = ',max_age)

max_active = df['active'].quantile(0.90)
print('Maximum range for active = ',max_active)


# In[39]:


def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['expenditure'] = max_value(df3, 'expenditure', max_expn)
    df3['age'] = max_value(df3, 'age', max_age)
    df3['active'] = max_value(df3, 'active', max_expn)
    df3['months'] = max_value(df3, 'months', max_months)


# In[40]:


X_train.expenditure.max(), X_test.expenditure.max()


# In[41]:


X_train.age.max(), X_test.age.max()


# In[42]:


X_train.active.max(), X_test.active.max()


# In[43]:


X_train.months.max(), X_test.months.max()


# In[44]:


# Enncode Categorical Values
X_train[categorical].head()


# In[45]:


#Encode Categorical Values 
import category_encoders as ce
encoder_owner = ce.BinaryEncoder(cols=['owner'])
X_train = encoder_owner.fit_transform(X_train)
X_test = encoder_owner.transform(X_test)

encoder_selfemp = ce.BinaryEncoder(cols=['selfemp'])
X_train = encoder_selfemp.fit_transform(X_train)
X_test = encoder_selfemp.transform(X_test)


# In[46]:


X_train.head()


# In[47]:


X_test.head()


# In[48]:


X_train.describe()


# In[49]:


cols = X_train.columns


# In[50]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[51]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[52]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[53]:


X_train.describe()


# In[54]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)


# In[55]:


# Predicted Results
y_pred_test = logreg.predict(X_test)

y_pred_test


# In[56]:


# probability of getting output as 0 - not approved
logreg.predict_proba(X_test)[:,0]


# In[57]:


# probability of getting output as 1 - Approved

logreg.predict_proba(X_test)[:,1]


# In[58]:


# accuracy Test
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# In[59]:


# Compare the train-set and test-set accuracy
y_pred_train = logreg.predict(X_train)
y_pred_train


# In[60]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[61]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))


# In[62]:


# check class distribution in test set

y_test.value_counts()


# In[63]:


# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# In[64]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[65]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

