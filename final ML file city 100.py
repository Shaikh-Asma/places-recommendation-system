#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('Review_db.csv')
data.head()


# In[3]:


data1=pd.read_csv('Users.csv')
data1.head()


# In[4]:


data2=pd.read_csv('Ratings.csv')
data2.head()


# In[5]:


data=pd.read_csv('Review_db.csv')
data2=pd.read_csv('Ratings.csv')
df=pd.concat([data, data2],axis = 1)
df.head()


# In[69]:


df=df.drop(['Date'],axis=1)


# In[70]:


df=df.drop(['Name'],axis=1)


# In[71]:


df=df.drop(['ISBN'],axis=1)


# In[72]:


new=df.drop(['Book-Rating'],axis=1)


# In[73]:


new.head()


# In[11]:


new.isnull().sum()


# In[74]:


new.duplicated().sum()


# In[75]:


new.drop_duplicates(inplace=True)


# In[76]:


new.duplicated().sum()


# In[77]:


new.shape


# # popularity based recommender system

# In[78]:


num_rating_df=new.groupby('City').count()['Rating'].reset_index()
num_rating_df.rename(columns={'Rating':'num_ratings'},inplace=True)
num_rating_df


# In[79]:


avg_rating_df=new.groupby('City').mean()['Rating'].reset_index()
avg_rating_df.rename(columns={'Rating':'avg_ratings'},inplace=True)
avg_rating_df


# In[80]:


popular_df=num_rating_df.merge(avg_rating_df,on='City')
popular_df


# In[81]:


popular_df=popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(100)
popular_df


# In[82]:


popular_df.merge(new,on='City').drop_duplicates('City')


# In[83]:


popular_df.merge(new,on='City').drop_duplicates('City').shape


# In[84]:


popular_df=popular_df.merge(new,on='City').drop_duplicates('City')[['Place','City','Review','Rating','num_ratings','avg_ratings']]


# In[85]:


popular_df


# # collaborative filtering

# In[86]:


x=new.groupby('User-ID').count()['Rating']>200
p=x[x].index


# In[87]:


filtered_rating=new[new['User-ID'].isin(p)]


# In[88]:


filtered_rating


# In[89]:


y=filtered_rating.groupby('City').count()['Rating']>=50
famous_places=y[y].index


# In[90]:


famous_places


# In[91]:


final_ratings=filtered_rating[filtered_rating['City'].isin(famous_places)]
final_ratings


# In[92]:


pt=final_ratings.pivot_table(index='City',columns='User-ID',values='Rating')


# In[93]:


pt.fillna(0,inplace=True)
pt


# In[94]:


from sklearn.metrics.pairwise import cosine_similarity


# In[95]:


similarity_score=cosine_similarity(pt)


# In[96]:


similarity_score.shape


# In[97]:


def recommend(Place):
    index=np.where(pt.index==Place)[0][0]
    similar_items=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data=[]
    for i in similar_items:
        item=[]
        temp_df = new[new['City'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('City')['Place'].values))
        item.extend(list(temp_df.drop_duplicates('City')['Review'].values))
        item.extend(list(temp_df.drop_duplicates('City')['Rating'].values))
        data.append(item)
    return data  


# In[98]:


recommend('New Delhi')


# In[99]:


np.where(pt.index=='Mumbai')[0][0]


# In[100]:


popular_df1=popular_df
popular_df1


# In[101]:


import pickle
pickle.dump(popular_df1,open('popular1.pkl','wb'))


# In[103]:


pt1=pt
pt1


# In[106]:


new1=new
new1=new1.head(10000)
new1


# In[107]:


similarity_score1=similarity_score
similarity_score1


# In[108]:


pickle.dump(pt1,open('pt1.pkl','wb'))
pickle.dump(new1,open('new1.pkl','wb'))
pickle.dump(similarity_score1,open('similarity_score1.pkl','wb'))


# In[109]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[110]:


from sklearn.preprocessing import LabelEncoder


# In[111]:


label_encoder=LabelEncoder()


# In[115]:


popular_df1['Place']=label_encoder.fit_transform(popular_df1['Place'])
popular_df1['City']=label_encoder.fit_transform(popular_df1['City'])
popular_df1['Review']=label_encoder.fit_transform(popular_df1['Review'])


# In[116]:


popular_df1


# In[117]:


x=popular_df1.iloc[:,0:3]
x


# In[118]:


y=popular_df1.iloc[:,-3]
y


# In[119]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[120]:


lr=LinearRegression()
lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
ypred


# In[121]:


lr.score(x_train,y_train)


# In[122]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[123]:


r2=r2_score(y_test,ypred)
r2


# In[124]:


m=mean_absolute_error(y_test,ypred)
m


# In[125]:


rmse=mean_squared_error(y_test,ypred)
np.sqrt(rmse)


# In[126]:


df1=pd.DataFrame({'actual':y_test,'predicted':ypred})
df1


# In[ ]:




