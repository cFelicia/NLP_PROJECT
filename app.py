import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

# Load NLTK resources if not done
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open('../ref/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('../ref/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define positive and negative words for the word cloud
positive_text = """
    movie,love,film,drama,plot, amazing, excellent, wonderful, incredible, like,fantastic, great, superb, enjoyable, thrilling, fun, delightful,
    hilarious, inspiring, heartfelt, charming, beautiful, excellent, perfect, breathtaking, lovely, masterpiece
"""

negative_text = """
    hate,movie,film, worst,terrible,drama,plot, awful, boring, bad, disappointing, dull, frustrating, poor, pathetic, sad, unfunny,
    awful, dreadful, pointless, bland, cringe, nonsensical, tiresome, unbearable
"""


# Text preprocessing function
def text_preprocessing(text, apply_stemming=True, apply_lemmatization=True):
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\bbr\b', '', text)
    
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    negation_words = {'no', 'not',"don't","can't", "isn't", "mustn't", 'hasn', 'shan', 'mustn', 'neither', 'ain', 'haven', 'none', "hadn't", 'hadn', "haven't", 'wouldn', "shouldn't", 'didn', "couldn't", "didn't", "wasn't", "shan't", "aren't", 'isn', 'needn', 'weren', 'mightn', "weren't", 'shouldn', "won't", 'never', "hasn't", "needn't", 'nor', 'cannot', 'couldn', 'doesn', 'wasn', "mightn't", "wouldn't", "doesn't", 'aren', 'won'}
    stop_words = stop_words.difference(negation_words)
    filtered_words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    processed_words = filtered_words
    
    if apply_stemming:
        processed_words = [stemmer.stem(word) for word in processed_words]
    if apply_lemmatization:
        processed_words = [lemmatizer.lemmatize(word) for word in processed_words]
    
    return ' '.join(processed_words)

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    st.pyplot(plt)

# Streamlit UI
st.title("🎬 Sentiment Analysis on Movie Reviews 🎬")
st.markdown("**Enter a movie review below, and let us predict whether it's positive or negative!**")

# Text input from the user
user_input = st.text_area("📝 Enter your movie review:")

# Predict the sentiment of the review
if st.button("🔍 Predict Sentiment"):
    if user_input:
        # Preprocess the input
        preprocessed_input = text_preprocessing(user_input)

        # Vectorize the input
        input_vector = vectorizer.transform([preprocessed_input])

        # Make prediction
        prediction = model.predict(input_vector)[0]
        
        # Display result
        if prediction == 1:
            st.success("The review is **Positive**! 😊")
            generate_wordcloud(positive_text)
        else:
            st.error("The review is **Negative**. 😔")
            generate_wordcloud(negative_text)
    else:
        st.warning("⚠️ Please enter a review before predicting!")
