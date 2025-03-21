import os  
import nltk  
import requests  
import pickle  
import pandas as pd  
from transformers import pipeline  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.pipeline import Pipeline  
import streamlit as st  

# Ensure NLTK data directory is created and added to the path  
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")  
os.makedirs(nltk_data_path, exist_ok=True)  
nltk.data.path.append(nltk_data_path)  



def download_nltk_resources():  
    """Download required NLTK resources."""  
    try:  
        nltk.data.find("tokenizers/punkt")  
    except LookupError:  
        nltk.download("punkt", download_dir=nltk_data_path)  
    try:  
        nltk.data.find("corpora/stopwords")  
    except LookupError:  
        nltk.download("stopwords", download_dir=nltk_data_path)  

# Call this function at the beginning of your Streamlit script before using NLTK  
download_nltk_resources()  

# Load sentiment analysis model  
sentiment_pipeline = pipeline("sentiment-analysis")  

# TMDb API Key (Replace with your own)  
API_KEY = "5fec0affb3191a8e860dd4017044c0f8"  

# Function to extract mood using sentiment analysis  
def extract_mood(prompt):  
    """Extracts mood using sentiment analysis"""  
    result = sentiment_pipeline(prompt)[0]  
    return result["label"]  

# Function to extract keywords from user input  
def extract_keywords(prompt):  
    """Extracts keywords to determine genre and meaning"""  
    words = word_tokenize(prompt.lower())  
    stop_words = set(stopwords.words("english"))  
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]  
    return filtered_words  

# Function to fetch movies by genre from TMDb API  
def get_movies_by_genre(genre):  
    """Fetches movies based on genre from TMDb API"""  
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre}"  
    response = requests.get(url)  
    if response.status_code == 200:  
        movies = response.json().get("results", [])  
        return movies[:10]  
    return []  

# Train and save a simple genre classification model  
def train_genre_classifier():  
    """Trains and saves a simple text classification model"""  
    data = {  
        "prompt": ["I want a scary horror movie", "A fun animated movie", "A thrilling action-packed film"],  
        "genre": ["Horror", "Animation", "Action"]  
    }  
    df = pd.DataFrame(data)  
    X = df["prompt"]  
    y = df["genre"]  
    pipeline = Pipeline([  
        ("tfidf", TfidfVectorizer()),  
        ("clf", LogisticRegression())  
    ])  
    pipeline.fit(X, y)  
    with open("genre_model.pkl", "wb") as f:  
        pickle.dump(pipeline, f)  

# Function to predict genre using trained model  
def predict_genre(prompt):  
    """Predicts genre from user input"""  
    with open("genre_model.pkl", "rb") as f:  
        model = pickle.load(f)  
    return model.predict([prompt])[0]  

# Train model if not present  
if not os.path.exists("genre_model.pkl"):  
    train_genre_classifier()  

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
