#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyforest
import numpy as np 
import seaborn as sns 
import matplotlib.pylab as plt 


# In[2]:


df=pd.read_csv('car data.csv')


# In[3]:


df.head(-1)


# In[4]:


df.shape


# In[5]:


print(df["Seller_Type"].unique(), df["Transmission"].unique(), df["Owner"].unique(),df["Fuel_Type"].unique())


# In[6]:


fdf=df[["Year","Selling_Price","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"]]


# In[7]:


dict = {'Year':[2017,2018,2015,1995,1852],
        'Selling_Price':[60,80,45,30,100],
        'Present_Price':[100,152,65,85,120],
        "Kms_Driven":[52000,56000,45000,8500,780000],
        "Fuel_Type":["Petrol","Petrol","CNG","Petrol","Diesel"],
        "Seller_Type":["Dealer","Dealer","Dealer","Dealer","Individual"],
        "Transmission":["Manual","Manual","Manual","Manual","Automatic"],
        "Owner":[1,1,3,1,0]
       }
df2 = pd.DataFrame(dict)
df2.head()


# In[8]:


fdf = fdf.append(df2, ignore_index = True)


# In[9]:


fdf.head()


# In[10]:


fdf.shape


# In[11]:


fdf.loc[fdf['Selling_Price'] == 60]


# In[12]:


fdf["Current_Year"]=2021


# In[13]:


fdf.tail()


# In[14]:


fdf["Total_Years"]=fdf["Current_Year"]-fdf["Year"]


# In[15]:


fdf.head(-1)


# In[16]:


fdf.drop(["Year","Current_Year"],axis=1, inplace=True)


# In[17]:


fdf=pd.get_dummies(fdf,drop_first=True) # three columns with categorial data namely Fuel_type,Seller type,Transmission
fdf.head()


# In[18]:


fdf.corr()


# In[19]:


sns.pairplot(fdf)


# In[20]:


corrmat=fdf.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(10,10)) #plot heatmap
sns.heatmap(fdf[top_corr_features].corr(), annot=True, fmt=".2f", cmap='Blues')


# In[21]:


fdf.head() #selling price is dependent feature


# In[22]:


X=fdf.iloc[:,1:]
Y=fdf.iloc[:,0]
X.head()


# In[23]:


Y.head()


# In[24]:


#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,Y)


# In[25]:


print(model.feature_importances_) #we can check with index values for the impotant feature


# In[26]:


#To visualize the most important feature
feature_importance=pd.Series(model.feature_importances_ , index=X.columns)
feature_importance.nlargest().plot(kind="bar")
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.25)


# In[28]:


X_train.shape


# In[29]:


Y_train.shape


# In[30]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[31]:


#hyperparameters 
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200)] 
print(n_estimators)


# In[32]:


from sklearn.model_selection import RandomizedSearchCV


# In[33]:


n_estimators=[int(x) for x in np.linspace(start=100,stop=1000,num=15)] 
max_features=["auto",'sqrt']
max_depth=[int(x) for x in np.linspace(5,30,num=6)] 
min_samples_split=[2,5,15,20,100,45,126]
min_samples_leaf=[2,5,10,20]


# In[34]:


random_grid={'n_estimators':n_estimators,
            'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf
            }
print(random_grid)


# In[35]:


rf=RandomForestRegressor()


# In[36]:


rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,random_state=45,n_jobs=1)


# In[37]:


rf_random.fit(X_train,Y_train)


# In[38]:


rf.get_params()


# In[39]:


predict=rf_random.predict(X_test)


# In[40]:


predict


# In[41]:


sns.histplot(Y_test-predict)


# In[42]:


plt.scatter(Y_test,predict)


# In[43]:


import pickle
file=open('random_forest-r_model.pkl','wb')
pickle.dump(rf_random,file) #dump


# In[ ]:




