# %%
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

# Sentiment Analysis using a pre-trained transformer model
sentiment_pipeline = pipeline("sentiment-analysis")

def extract_mood(prompt):
    """Extracts mood using sentiment analysis"""
    result = sentiment_pipeline(prompt)[0]
    return result["label"]

def extract_keywords(prompt):
    """Extracts keywords from the user input to determine genre and meaning."""
    words = word_tokenize(prompt.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words("english")]
    return filtered_words

# Example usage
if __name__ == "__main__":
    prompt = "I want to watch a happy comedy movie with some action!"
    mood = extract_mood(prompt)
    keywords = extract_keywords(prompt)
    print(f"Mood: {mood}")
    print(f"Keywords: {keywords}")


# %%
import requests

API_KEY = "your_tmdb_api_key"  # Replace with your TMDb API Key

def get_movies_by_genre(genre):
    """Fetches movies based on genre from TMDb API"""
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre}"
    response = requests.get(url)
    if response.status_code == 200:
        movies = response.json().get("results", [])
        return movies[:10]  # Return top 10 movies
    return []

def search_movies(query):
    """Searches for movies based on user query"""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
    response = requests.get(url)
    if response.status_code == 200:
        movies = response.json().get("results", [])
        return movies[:5]  # Return top 5 matches
    return []

# Example usage
if __name__ == "__main__":
    genre = "Action"
    movies = get_movies_by_genre(genre)
    print(f"Top movies in {genre}:")
    for movie in movies:
        print(movie["title"])


# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Sample dataset (expand this with real data)
data = {
    "prompt": ["I want a scary horror movie", "A fun animated movie", "A thrilling action-packed film"],
    "genre": ["Horror", "Animation", "Action"]
}

df = pd.DataFrame(data)

# Training a simple model
X = df["prompt"]
y = df["genre"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

pipeline.fit(X, y)

# Save model
with open("genre_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Function to predict genre
def predict_genre(prompt):
    with open("genre_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict([prompt])[0]

# Example usage
if __name__ == "__main__":
    user_prompt = "I want a fun animated movie"
    predicted_genre = predict_genre(user_prompt)
    print(f"Predicted Genre: {predicted_genre}")


# %%
import streamlit as st
from mood_extractor import extract_mood, extract_keywords
from fetch_movies import get_movies_by_genre
from genre_classifier import predict_genre

st.title("🎬 Movie Recommendation System")
st.write("Enter a description of the movie you feel like watching, and we'll recommend the best match!")

# User input
user_input = st.text_area("Enter your movie preference (e.g., 'I want a happy comedy movie with some action!')")

if st.button("Get Recommendations"):
    if user_input:
        # Extract mood, keywords, and genre
        mood = extract_mood(user_input)
        keywords = extract_keywords(user_input)
        genre = predict_genre(user_input)

        # Fetch movie recommendations
        movies = get_movies_by_genre(genre)

        # Display results
        st.subheader(f"Mood: {mood}")
        st.subheader(f"Predicted Genre: {genre}")
        
        if movies:
            st.subheader("🎥 Recommended Movies:")
            for movie in movies:
                st.write(f"**{movie['title']}** (⭐ {movie['vote_average']})")
        else:
            st.write("No recommendations found. Try a different prompt!")

    else:
        st.warning("Please enter a movie preference.")




