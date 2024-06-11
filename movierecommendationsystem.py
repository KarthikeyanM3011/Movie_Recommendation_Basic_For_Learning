import numpy as np
import pandas as pd
import streamlit as st
#To find similar value in case of any spelling mistakes
import difflib
#To easily classify the data based on similarities in their numerical value
from sklearn.metrics.pairwise import cosine_similarity
#convert all text to numerical values to classify easily
from sklearn.feature_extraction.text import TfidfVectorizer
st.header("Movie Recommendation System")
try:
  data=pd.read_csv("C:\\Users\\Administrator\\Downloads\\movies.csv")
  movies_tit=data['title'].values
  movie_name=st.selectbox("select the movies",movies_tit)
  selected_data=['genres','keywords','tagline','cast','director']

  #replacing the null values with a null string
  for s_data in selected_data:
    data[s_data]=data[s_data].fillna('')
    
  #combine all data to convert them into numerical value
  combined_data=data['genres']+' '+data['keywords']+' '+data['tagline']+' '+data['cast']+' '+data['director']

  #converting into vectors that is numerical values
  vectorizer=TfidfVectorizer()
  v_data=vectorizer.fit_transform(combined_data)

  #based on the above numerical values plotting each data to its similar data 
  similar_data=cosine_similarity(v_data)

  #getting all the movies name from the data list
  movie_list=data['title'].tolist()

  #finding similar matches of movie name in data set to that of user input
  matches = difflib.get_close_matches(movie_name,movie_list)

  #finding the index of most matching movie given by user
  index = data[data.title==matches[0]]['index'].values[0]

  #getting the nearest similar movies by cosine_similarity
  similar_movies=list(enumerate(similar_data[index]))

  #sorting them based on the value in index 1 
  similar_movie=sorted(similar_movies, key=lambda x: x[1], reverse=True)

  #printing the top 25 moies recommended similar movies based on the user input
  print("Movies recommended for you are : \n")
  i=0
  lis=[]
  lis_rate=[]
  for movies in similar_movie:
    movie_title=data[data.index==movies[0]]['title'].values[0]
    movie_vote=data[data.index==movies[0]]['vote_average'].values[0]
    
    if(i<=5):
      lis.append(movie_title)
      lis_rate.append(movie_vote)
    else:
      break
    i+=1
except Exception as e:
  print(e)
  
if st.button("Search"):
  col1,col2=st.columns(2)
  with col1:
    st.title("Title")
    st.write(lis[0])
    st.write(lis[1])
    st.write(lis[2])
    st.write(lis[3])
    st.write(lis[4])
    
  with col2:
    st.title("Rating")
    st.write(lis_rate[0])
    st.write(lis_rate[1])
    st.write(lis_rate[2])
    st.write(lis_rate[3])
    st.write(lis_rate[4])
    