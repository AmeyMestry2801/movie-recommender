import pandas as pd

# Load movies
movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', encoding='ISO-8859-1',
                     names=['movieId', 'title', 'genres'])

# Load ratings
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python', encoding='ISO-8859-1',
                      names=['userId', 'movieId', 'rating', 'timestamp'])

print("Movies sample:")
print(movies.head())

print("\nRatings sample:")
print(ratings.head())

# Preprocess the genres column
movies['genres'] = movies['genres'].str.split('|')

# Confirm it's working
print("\nGenres after splitting:")
print(movies['genres'].head())


from sklearn.preprocessing import MultiLabelBinarizer

# Create the encoder
mlb = MultiLabelBinarizer()

# Transform genres into one-hot encoded DataFrame
genre_matrix = pd.DataFrame(
    mlb.fit_transform(movies['genres']),
    columns=mlb.classes_,
    index=movies.index
)

# Optional: print first few rows
print("\nOne-hot encoded genre matrix:")
print(genre_matrix.head())


from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all movies
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Optional: show similarity of first movie with all others
print("\nCosine similarity with first movie:")
print(cosine_sim[0])


def recommend_movies(title, movies=movies, cosine_sim=cosine_sim, top_n=10):
    # Get index of the movie that matches the title
    idx = movies[movies['title'] == title].index

    if len(idx) == 0:
        return ["❌ Movie not found in database."]
    
    idx = idx[0]

    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top_n similar movies (excluding itself)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    # Return recommended movie titles
    return movies['title'].iloc[movie_indices].tolist()
print("\nRecommendations for 'Toy Story (1995)':")
recommendations = recommend_movies('Toy Story (1995)')
for movie in recommendations:
    print(f"🎬 {movie}")
