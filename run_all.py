import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("--- Starting NLP Pipeline ---")

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])
    print(f"Data loaded: {len(df)} rows.")
except FileNotFoundError:
    print("Error: train.txt not found. Please ensure it is in the project directory.")
    exit(1)

# 2. Preprocessing
print("Preprocessing data...")
unique_emotions = df['emotion'].unique()
emotion_numbers = {emo: i for i, emo in enumerate(unique_emotions)}
df['emotion'] = df['emotion'].map(emotion_numbers)

df['text'] = df['text'].apply(lambda x: str(x).lower())

def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

df['text'] = df['text'].apply(remove_punc)

def remove_numbers(txt):
    return "".join([i for i in txt if not i.isdigit()])

df['text'] = df['text'].apply(remove_numbers)

def remove_emojis(txt):
    return "".join([i for i in txt if i.isascii()])

df['text'] = df['text'].apply(remove_emojis)

# NLTK Setup
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def remove_stopwords(txt):
    return ' '.join([word for word in txt.split() if word not in stop_words])

df['text'] = df['text'].apply(remove_stopwords)
print("Preprocessing complete.")

# 3. Train/Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.20, random_state=42)

# 4. Feature Extraction & Modeling
print("Training models...")

# Bag of Words + Naive Bayes
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
pred_bow = nb_model.predict(X_test_bow)
print(f"Bag of Words Accuracy: {accuracy_score(y_test, pred_bow):.4f}")

# TF-IDF + Naive Bayes
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
nb2_model = MultinomialNB()
nb2_model.fit(X_train_tfidf, y_train)
y_pred_nb2 = nb2_model.predict(X_test_tfidf)
print(f"TF-IDF Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb2):.4f}")

# TF-IDF + Logistic Regression (Selected Best Model)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)
log_pred = logistic_model.predict(X_test_tfidf)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred):.4f}")

# 5. Saving Models
print("Saving model and vectorizer...")
# We save the logistic_model and tfidf_vectorizer as they gave better accuracy
pickle.dump(logistic_model, open("model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))

print("--- Pipeline Completed Successfully ---")
print("Files saved: model.pkl, vectorizer.pkl")
