# 🎬 Movie Review Sentiment Analysis using BoW

This repository hosts an NLP project that uses BoW to classify movie reviews as positive or negative based on sentiment. Built with a logistic reggression, 
the model is trained on the IMDB dataset  from Kaggle and deployed using Streamlit, providing a user-friendly interface for sentiment prediction.

## 📂 Project Structure
- **`data/`**: Contains the compressed IMDB dataset (`IMDB_Dataset.zip`).
-  **`preprocessing.py`**: Includes text preprocessing functions such as stopword removal, negation handling, and BoW embeddings.
- **`ref/`**: Stores the trained RNN model and vectorizer files (compressed for GitHub compatibility).
- **`app.py`**: Streamlit app that enables users to input a movie review and receive sentiment predictions.
- **`model.py`**: Script for building, training, and evaluating the RNN model.

---

## 🚀 Features
1. **Sentiment Analysis**: Classifies reviews as positive or negative.
2. **Text Preprocessing**: Handles punctuation, stopwords, and negations, and tokenizes text for input into the model.
3. **Word Embeddings**: Uses Word2Vec to create numerical representations of words for the RNN.
4. **Streamlit Deployment**: An interactive web app for real-time sentiment prediction with word clouds for visual insights.

---



## 🏃 Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Use the Web App**:
   - Open the provided URL in your browser.
   - Enter a movie review and hit "Predict Sentiment."
   - View the sentiment prediction and a word cloud representing common words in positive or negative reviews.

---

## 🧰 Code Overview

- **Data Loading & Preprocessing**:
  - `preprocessing.py`: Functions to clean and tokenize text, handling negations and generating word embeddings with Word2Vec.

- **Model Training**:
  - `model.py`: RNN model built with Keras, trained to identify sentiment patterns in the IMDB dataset.

- **Streamlit App**:
  - `app.py`: Hosts the web interface and integrates text preprocessing, model inference, and word cloud visualizations.

---

## 📈 Model Details

- **Architecture**: RNN with LSTM layers for sequence analysis.
- **Training**: Trained on IMDB movie reviews to predict positive or negative sentiment.
- **Evaluation**: Model tested for accuracy, ensuring balanced performance across positive and negative samples.

---

## 🔮 Future Enhancements
- Fine-tuning the RNN for improved accuracy.
- Supporting more granular emotional analysis.
- Adding feedback mechanisms for continuous improvement.

---

## 🙏 Acknowledgments

- **IMDB** for the dataset.
- **NLTK and Word2Vec** for text processing and embedding.
- **Streamlit** for enabling rapid deployment of the app.

---

Feel free to contribute by opening issues or submitting pull requests. Enjoy using the sentiment analysis tool!

--- 

This `README.md` is organized for clarity and covers all the major aspects of your project, making it easy for others to understand and use. Let me know if you need further adjustments!
