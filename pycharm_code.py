import pandas as pd
import streamlit as st
import pickle
import requests
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=ce7b0f74d78219ddfda3207f2386dd01&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    dis=sim[movie_index]
    movie_list=sorted(list(enumerate(dis)),reverse=True ,key=lambda x:x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in movie_list[1:6]:
        movie_id = movies.iloc[i[0]].id
        #fetch posters from API
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return  recommended_movie_names , recommended_movie_posters
st.title("Movie Recommendation system")
movie_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movie_dict)
sim=pickle.load(open('sim.pkl','rb'))

option = st.selectbox(
    "Type or select a movie from the dropdown",
    (movies['title'].values))


if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters = recommend(option)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])




