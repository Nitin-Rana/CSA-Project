#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


titanic_data = pd.read_csv('titanic.csv')


# In[24]:


titanic_data.head(10)


# In[25]:


print('Number of data is: ' + str(len(titanic_data.index)))


# # Analysing Data

# In[26]:


sns.countplot('Survived', data=titanic_data)


# In[27]:


sns.countplot('Survived', hue='Sex', data=titanic_data)


# In[28]:


sns.countplot('Survived', hue='Pclass', data=titanic_data)


# In[29]:


titanic_data['Age'].plot.hist()


# In[30]:


titanic_data['Fare'].plot.hist()


# In[31]:


titanic_data.info()


# In[32]:


sns.countplot('SibSp', data=titanic_data)


# # Data Wrangling

# In[33]:


titanic_data.isnull()


# In[34]:


titanic_data.isnull().sum()


# In[35]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap='YlGnBu')


# In[36]:


sns.boxplot(x='Pclass', y='Age', data=titanic_data)


# In[37]:


titanic_data.head(5)


# In[38]:


titanic_data.drop('Cabin', axis=1, inplace=True)


# In[39]:


titanic_data.head(5)


# In[40]:


titanic_data.dropna(inplace=True)


# In[41]:


sns.heatmap(titanic_data.isnull(), yticklabels=False)


# In[42]:


titanic_data.isnull().sum()


# In[43]:


sex =pd.get_dummies(titanic_data['Sex'], drop_first=True)


# In[44]:


sex.head(5)


# In[45]:


embark = pd.get_dummies(titanic_data['Embarked'],drop_first=True)


# In[46]:


embark.head(5)


# In[47]:


Pcl = pd.get_dummies(titanic_data['Pclass'], drop_first=True)


# In[48]:


Pcl.head(5)


# In[49]:


titanic_data = pd.concat([titanic_data,sex,embark,Pcl], axis=1)


# In[50]:


titanic_data.head(5)


# In[51]:


titanic_data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis =1, inplace=True)


# In[52]:


titanic_data.head(5)


# # Train Data

# In[53]:


from sklearn.model_selection import train_test_split


# In[59]:


X = titanic_data.drop('Survived', axis=1)
X.head()
y = titanic_data['Survived']


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


logmodel = LogisticRegression()


# In[ ]:





# In[66]:


logmodel.fit(X_train, y_train)


# In[67]:


predictions = logmodel.predict(X_test)


# In[70]:


from sklearn.metrics import classification_report


# In[71]:


classification_report(y_test,predictions)


# In[72]:


from sklearn.metrics import confusion_matrix


# In[79]:


confusion_matrix(y_test, predictions)


# # Accuracy of Algorithm
# 

# In[77]:


from sklearn.metrics import accuracy_score


# In[78]:


accuracy_score(y_test, predictions)


# In[ ]:




