# NLP_EMOTION_DETECTOR
NLP-based Emotion Detection system that predicts whether a given text is Positive, Neutral, or Negative with ~90% accuracy. Built using Scikit-learn and deployed using Streamlit for real-time interaction.
# 🧠 Real-Time Emotion Detection using NLP

An AI-powered web application that analyzes user input text and predicts the underlying emotion — **Positive 😊, Neutral 😐, or Negative 😢** — with an accuracy of approximately **90%**.

This project demonstrates an end-to-end **Natural Language Processing (NLP)** pipeline, from data preprocessing and model training to real-time deployment using an interactive web interface.

---

## 🚀 Demo

> 💡 Enter any sentence and instantly get emotion prediction with confidence score.

---

## ✨ Features

* 🔍 Real-time emotion detection
* 📊 ~90% model accuracy
* ⚡ Fast and lightweight prediction
* 🎯 Confidence score output
* 💻 Interactive UI using Streamlit
* 🔁 End-to-end ML pipeline

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* Streamlit

---

## 🧠 How It Works

1. **Text Preprocessing**

   * Tokenization
   * Stopword removal
   * Stemming/Lemmatization

2. **Feature Extraction**

   * TF-IDF / Count Vectorization

3. **Model Training**

   * Machine Learning classification model

4. **Prediction**

   * Input text → Vectorized → Model → Emotion Output

---

## 📁 Project Structure

```
emotion-app/
│
├── app.py                # Streamlit UI
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # Text vectorizer
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Example

| Input Text          | Prediction  |
| ------------------- | ----------- |
| I love this product | 😄 Positive |
| It’s okay           | 😐 Neutral  |
| I hate this         | 😢 Negative |

---

## 📦 Requirements

You can generate this using:

```bash
pip freeze > requirements.txt
```

Or manually include:

```
streamlit
scikit-learn
pandas
numpy
nltk
```

---

## 🌍 Deployment

This project can be easily deployed using:

* Streamlit Community Cloud
* Render
* Heroku (optional)

---

## 🚀 Future Improvements

* 🔥 Multi-emotion classification (happy, sad, angry, etc.)
* 🤖 Transformer-based models (BERT)
* 🎤 Voice input support
* 💬 Chatbot integration
* 🌐 REST API for external usage

---

## 💼 Use Cases

* Sentiment analysis
* Social media monitoring
* Customer feedback analysis
* Chatbots & virtual assistants

---

