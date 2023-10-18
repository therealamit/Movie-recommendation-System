#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")
#Merging of data sets
movies=movies.merge(credits ,on='title')

#Imporant columns
#genre
#id
#keywords
#title
#overview
#cast
#crew



# In[8]:


movies


# In[52]:


movies=movies[['id','title','overview','genres','keywords','cast','crew']]
movies=movies.dropna()


# In[53]:


import ast
def convert(obj):
 L=[]
 for i in ast.literal_eval(obj):
   L.append(i['name'])
 return L    


# In[54]:


def convert1(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
         L.append(i['name'])
         counter+=1
        else:
            break
    return L; 


# In[55]:


def convert2(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director" :
            L.append(i['name'])
            break
    return L;   


# In[35]:


movies


# In[56]:


movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies['cast']=movies['cast'].apply(convert1)
movies['crew']=movies['crew'].apply(convert2)
movies['overview']=movies['overview'].apply(lambda x:x.split())
# removing spaced between them to make them unique key
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
movn=movies[['id','title','tags']]


# In[59]:


movn['tags']=movn['tags'].apply(lambda x:" ".join(x))


# movn['tags']=movn['tags'].apply(lambda x:x.lower())

# In[62]:


movn['tags']=movn['tags'].apply(lambda x:x.lower())


# In[63]:


movn


# In[171]:


movies.size


# In[6]:


credits.shape


# In[7]:


movies.shape


# In[172]:


movies.head()


# In[18]:


#Merging of data sets
movies=movies.merge(credits ,on='title')


# In[19]:


movies.info()


# In[43]:


#Imporant columns
#genre
#id
#keywords
#title
#overview
#cast
#crew
movies=movies[['id','title','overview','genres','keywords','cast_x','crew_x']]


# In[44]:


movies.head()


# In[46]:


movie_d.head()


# In[47]:


movies.isnull().sum()


# In[ ]:


#here we can see overview has three missing datas


# In[48]:


movie_d.duplicated().sum()


# In[57]:


movie_d.dropna(inplace=True)


# In[58]:


movies=movies.dropna()


# In[59]:


movies.isnull().sum()


# In[60]:


movie_d.duplicated().sum()


# In[61]:


movies.duplicated().sum()


# In[55]:


movies.duplicated().sum()


# In[173]:


movie_d.duplicated().sum()


# In[62]:


movies.isnull().sum()


# In[63]:


movies.duplicated().sum()


# In[64]:


movies.duplicated().sum()


# In[65]:


movies.head()


# In[68]:


movies.duplicated().sum()


# In[72]:


import ast
def convert(obj):
 L=[]
 for i in ast.literal_eval(obj):
   L.append(i['name'])
 return L    


# In[76]:


movies['genres']=movies['genres'].apply(convert)


# In[77]:


movies.head()


# In[78]:


movies['keywords'].apply(convert)


# In[79]:


movies['keywords']=movies['keywords'].apply(convert)


# In[80]:


movies.head()


# In[85]:


def convert1(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
         L.append(i['name'])
         counter+=1
        else:
            break
    return L;     


# In[98]:


def convert2(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director" :
            L.append(i['name'])
            break
    return L;   
          


# In[86]:


movies['cast_x'].apply(convert1)


# In[87]:


movies['cast_x']=movies['cast_x'].apply(convert1)


# In[106]:


movies.head()


# In[94]:


movies['crew_x']


# In[105]:


movies['crew_x']movies['crew_x'].apply(convert2)


# In[103]:


z


# In[107]:


movies.head()


# In[108]:


movie_d.head()
movie


# movies=movie_d

# In[109]:


movies


# In[110]:


movies.head()


# In[111]:


movie_d.head()


# In[112]:


movies=movie_d


# In[113]:


movie.shape()


# movies.head()

# In[114]:


movies.head()


# In[115]:


movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies['cast_x']=movies['cast_x'].apply(convert1)
movies['crew_x']=movies['crew_x'].apply(convert2)


# In[116]:


movies.head()


# In[117]:


movied=movies


# In[118]:


movied


# In[119]:


movies


# In[120]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[121]:


movies.head()


# In[124]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[126]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[127]:


movies['cast_x']=movies['cast_x'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[128]:


movies['crew_x']=movies['crew_x'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[129]:


movies.head()


# In[131]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast_x']+movies['crew_x']


# In[132]:


movies.head()


# In[153]:


movn=movies[['id','title','tags']]


# In[154]:


movn


# In[155]:


movn['tags'].apply(lambda x:" ".join(x))


# In[156]:


movn['tags'][0]


# In[38]:


movn['tags'].apply(lambda x:x.lower())


# In[39]:


movn['tags']=movn['tags'].apply(lambda x:" ".join(x))


# In[163]:


movn.head()


# In[164]:


movn['tags'][0]


# In[40]:


movn['tags'].apply(lambda x:x.lower())


# In[166]:


movn['tags']=movn['tags'].apply(lambda x:x.lower())


# In[167]:


movn.head()


# In[168]:


min.desc


# In[169]:


movn.describe()


# In[174]:


movn.head()


# In[64]:


#text vectorisation (bag of wrods ->)
from sklearn.feature_extraction.text import CountVectorizer


# In[65]:


cv=CountVectorizer(stop_words='english', max_features=5000)


# In[66]:


vectors=cv.fit_transform(movn['tags']).toarray()


# In[67]:


vectors


# In[68]:


cv.get_feature_names()


# In[69]:


#remove similar words like love,loved,loving to love only using nltk lib because it will make our data frame more accurate
#it is good practice to remove such words
import nltk


# In[70]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[71]:


def stem(text):
    y=[]
    for i in text.split():
     y.append(ps.stem(i))
    return " ".join(y)    


# In[72]:


movn['tags'].apply(stem)


# In[73]:


#Now we need to find distanced between each vector but we know that for higher dimension euclidean distance is not a good
#good measure so what we use instead of this is cosine angle which is the angle between two vectors..lesser the 
#angle less is the distance between them.
from sklearn.metrics.pairwise import cosine_similarity


# In[74]:


sim=cosine_similarity(vectors)


# In[75]:


def recommend(movie):
    movie_index=movn[movn['title']==movie].index[0]
    dis=sim[movie_index]
    movie_list=sorted(list(enumerate(dis)),reverse=True ,key=lambda x:x[1])
    for i in movie_list[1:10]:
        print(movn.iloc[i[0]].title)
    
    


# In[76]:


recommend('Avatar')


# In[77]:


movn.head(10)


# In[78]:


sim


# In[221]:


sim[539]


# In[79]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    
vector = cv.fit_transform(movn['tags']).toarray()
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


def recommend(movie):
    index = movn[movn['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:10]:
        print(movn.iloc[i[0]].title)
        
    


# In[80]:


recommend('Gandhi')


# In[232]:


recommend('Avatar')


# In[81]:


import pickle
pickle.dump(movn.to_dict(),open('movie_dict.pkl','wb'))


# In[82]:


pickle.dump(sim,open('sim.pkl','wb'))


# In[1]:


movn


# In[ ]:




