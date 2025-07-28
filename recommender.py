import numpy as np
import pandas as pd
import joblib
from lightfm import LightFM

# ==== Load encoders and model ====
user_encoder = joblib.load("models/user_encoder.pkl")
movie_encoder = joblib.load("models/movie_encoder.pkl")
model = joblib.load("models/lightfm_hybrid_model.pkl")
item_features = joblib.load("models/item_features_sparse.pkl")

# ==== Load data ====
movies = pd.read_json("indian movie data/cleaned_movies.json", lines=True)
ratings = pd.read_json("indian movie data/cleaned_ratings.json", lines=True)

# ==== Recommendation Function ====

def recommend_movies(user_id, top_n=5):
    if user_id not in user_encoder.classes_:
        return [], []

    user_idx = user_encoder.transform([user_id])[0] ## convert encoded index
    n_items = len(movie_encoder.classes_)

    # Predict scores
    user_ids = np.repeat(user_idx, n_items)
    item_ids = np.arange(n_items)
    scores = model.predict(user_ids=user_ids, item_ids=item_ids, item_features=item_features)

    # Watched movie IDs
    watched_ids = ratings[ratings['user_id'] == user_id]['movie_id'].unique()

    # Filter out already watched movies from top-N
    scores_filtered = [(i, s) for i, s in enumerate(scores) if movie_encoder.inverse_transform([i])[0] not in watched_ids]
    scores_filtered = sorted(scores_filtered, key=lambda x: x[1], reverse=True)[:top_n]
    top_indices = [i for i, _ in scores_filtered]
    top_movie_ids = movie_encoder.inverse_transform(top_indices)

    # Recommended movies
    top_movies = movies[movies['movie_id'].isin(top_movie_ids)][['name', 'genre', 'language']]

    # Watched movies
    watched_movies = movies[movies['movie_id'].isin(watched_ids)][['name', 'genre', 'language']]

    return top_movies.reset_index(drop=True), watched_movies.reset_index(drop=True)
