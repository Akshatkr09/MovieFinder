import streamlit as st
import os
import requests
import pandas as pd
import spacy
from transformers import pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure spaCy model is available
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# TMDb API Key
API_KEY = "5fec0affb3191a8e860dd4017044c0f8"  # Your provided API key

# Function to extract mood using sentiment analysis
def extract_mood(prompt):
    """Extracts mood using sentiment analysis"""
    result = sentiment_pipeline(prompt)[0]
    return result["label"]

# Function to extract keywords using spaCy
def extract_keywords(prompt):
    """Extracts keywords from user input using spaCy"""
    doc = nlp(prompt.lower())
    keywords = [token.text for token in doc if token.is_alpha and token.text not in STOP_WORDS]
    return keywords

# Function to fetch movies by genre from TMDb API
def get_movies_by_genre(genre):
    """Fetches movies based on genre from TMDb API"""
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre}"
    response = requests.get(url)
    if response.status_code == 200:
        movies = response.json().get("results", [])
        return movies[:10]
    return []

# Train a simple genre classification model in memory
def train_genre_classifier():
    """Trains and returns a simple text classification model"""
    data = {
        "prompt": ["I want a scary horror movie", "A fun animated movie", "A thrilling action-packed film"],
        "genre": ["Horror", "Animation", "Action"]
    }
    df = pd.DataFrame(data)
    X = df["prompt"]
    y = df["genre"]
    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])
    model_pipeline.fit(X, y)
    return model_pipeline

# Load the genre classifier model into memory
genre_classifier = train_genre_classifier()

# Function to predict genre using trained model
def predict_genre(prompt):
    """Predicts genre from user input"""
    return genre_classifier.predict([prompt])[0]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a description of the movie you feel like watching, and we'll recommend the best match!")

# User input
user_input = st.text_area("Enter your movie preference (e.g., 'I want a happy comedy movie with some action!')")

if st.button("Get Recommendations"):
    if user_input:
        mood = extract_mood(user_input)
        keywords = extract_keywords(user_input)
        genre = predict_genre(user_input)
        movies = get_movies_by_genre(genre)

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
