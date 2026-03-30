import streamlit as st
import pickle
st.markdown("""
<style>
textarea {
    font-size: 18px !important;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)
# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Emotion Detector", page_icon="🧠")

# Title
st.title("🧠 Real-Time Emotion Detection")
st.write("Type a sentence and detect its emotion instantly")

# Input box
user_input = st.text_area("Enter your sentence here:")

# Predict button
if st.button("Analyze Emotion"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    
    else:
        # Transform input
        transformed = vectorizer.transform([user_input])
        
        # Prediction
        prediction = model.predict(transformed)[0]
        
        # (Optional) Probability
        try:
            proba = model.predict_proba(transformed).max()
            confidence = round(proba * 100, 2)
        except:
            confidence = None
        
        # Map result
        emotion_map = {
            0: "😢 Sadness",
            1: "😠 Anger",
            2: "❤️ Love",
            3: "😲 Surprise",
            4: "😨 Fear",
            5: "😊 Joy"
        }
        result = emotion_map.get(prediction, "❓ Unknown")
        
        # Display result
        st.success(f"Result: {result}")
        
        if confidence:
            st.info(f"Confidence: {confidence}%")