import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data (reuse your existing code or load preprocessed files)
@st.cache_data
def load_data():
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', encoding='ISO-8859-1',
                         names=['movieId', 'title', 'genres'])
    movies['genres'] = movies['genres'].str.split('|')

    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(movies['genres']),
        columns=mlb.classes_,
        index=movies.index
    )
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    return movies, cosine_sim

movies, cosine_sim = load_data()

def recommend_movies(title, movies=movies, cosine_sim=cosine_sim, top_n=10):
    idx = movies[movies['title'] == title].index
    if len(idx) == 0:
        return ["Movie not found."]
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie you like:", movies['title'].tolist())

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    st.write(f"Movies similar to **{selected_movie}**:")
    for rec in recommendations:
        st.write(f"- {rec}")
