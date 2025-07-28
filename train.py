import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer  #LabelEncoder: Convert user/movie IDs to numeric, MultiLabelBinarizer: Convert genre/language lists to vectors
from scipy.sparse import coo_matrix, hstack
from lightfm import LightFM
import os

# ==== Load cleaned data ====
movies = pd.read_json("indian movie data/cleaned_movies.json", lines=True)
ratings = pd.read_json("indian movie data/cleaned_ratings.json", lines=True)

print("Loaded movies:", movies.shape)
print("Loaded ratings:", ratings.shape)

# ==== Encode user_id and movie_id ====
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['user_id'])
ratings['item'] = movie_encoder.fit_transform(ratings['movie_id'])

# Save encoders
os.makedirs("models", exist_ok=True)
joblib.dump(user_encoder, "models/user_encoder.pkl")
joblib.dump(movie_encoder, "models/movie_encoder.pkl")

# ==== Create interaction matrix ====
# user item interaction mtrix
interactions = coo_matrix(                                                   # stores only non-zero entries,
    (ratings['rating'].astype(float), (ratings['user'], ratings['item']))
)

print(" Interaction matrix shape:", interactions.shape)

# ==== Content Features (Genre, Language, etc.) ====

# Filter movies to include only those in ratings
rated_movie_ids = ratings['movie_id'].unique()
content_df = movies[movies['movie_id'].isin(rated_movie_ids)][['movie_id', 'genre', 'language']].copy()

# Map movie_id to encoded 'item' values using movie_encoder
content_df['item'] = movie_encoder.transform(content_df['movie_id'])

# Handle multi-hot encoding
def process_list_column(col):
    return col.apply(lambda x: x if isinstance(x, list) else [])

mlb_genre = MultiLabelBinarizer()
mlb_lang = MultiLabelBinarizer()

genre_encoded = mlb_genre.fit_transform(process_list_column(content_df['genre']))
lang_encoded = mlb_lang.fit_transform(process_list_column(content_df['language']))

## Combine features
features = np.hstack([genre_encoded, lang_encoded])

# Convert to CSR before sorting (FIXED)
features_sparse = coo_matrix(features).tocsr()

# Sort by 'item' so order matches with LightFM expectations
sorted_index = np.argsort(content_df['item'].values)
sorted_features = features_sparse[sorted_index]

print("Content features shape:", sorted_features.shape)

# ==== Train the Hybrid Model ====
model = LightFM(loss='warp', no_components=64)

print("Training model...")
model.fit(interactions, item_features=sorted_features, epochs=20, num_threads=4)
print("Training done!")

# ==== Save Model ====
joblib.dump(model, "models/lightfm_hybrid_model.pkl")
joblib.dump(sorted_features, "models/item_features_sparse.pkl")

print("Model and features saved successfully.")
