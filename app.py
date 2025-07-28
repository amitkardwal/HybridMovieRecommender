from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# ==== Load Encoders, Model, and Data ====
user_encoder = joblib.load("models/user_encoder.pkl")
movie_encoder = joblib.load("models/movie_encoder.pkl")
model = joblib.load("models/lightfm_hybrid_model.pkl")
item_features = joblib.load("models/item_features_sparse.pkl")

movies = pd.read_json("indian movie data/cleaned_movies.json", lines=True)
ratings = pd.read_json("indian movie data/cleaned_ratings.json", lines=True)

# ==== Recommendation Function ====
def recommend_movies(user_id, top_n=5):
    if user_id not in user_encoder.classes_:
        return None, None

    user_idx = user_encoder.transform([user_id])[0]
    n_items = len(movie_encoder.classes_)

    user_ids = np.repeat(user_idx, n_items)
    item_ids = np.arange(n_items)

    scores = model.predict(user_ids=user_ids, item_ids=item_ids, item_features=item_features)
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_movie_ids = movie_encoder.inverse_transform(top_indices)

    # Recommended movies
    recommended = movies[movies['movie_id'].isin(top_movie_ids)][['name', 'genre', 'language']].copy()
    recommended['genre'] = recommended['genre'].apply(lambda x: ', '.join(x))
    recommended['language'] = recommended['language'].apply(lambda x: ', '.join(x))

    # Watched movies
    watched_ids = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    watched = movies[movies['movie_id'].isin(watched_ids)][['name', 'genre', 'language']].copy()
    watched['genre'] = watched['genre'].apply(lambda x: ', '.join(x))
    watched['language'] = watched['language'].apply(lambda x: ', '.join(x))

    return recommended.to_dict(orient="records"), watched.to_dict(orient="records")

# ==== Flask App ====
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    watched = []
    message = ""
    user_id = ""

    if request.method == "POST":
        user_id = request.form.get("user_id").strip()
        top_n = request.form.get("top_n")

        try:
            top_n = int(top_n)
        except:
            top_n = 5

        result, watched_movies = recommend_movies(user_id, top_n)
        if result is None:
            message = f"❌ User ID '{user_id}' not found in training data."
        else:
            recommendations = result
            watched = watched_movies

    return render_template("index.html",
                           recommendations=recommendations,
                           watched=watched,
                           message=message,
                           user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)
