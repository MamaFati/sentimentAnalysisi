import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import spacy
from preprocessing import clean_text

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load models
logreg_model = joblib.load("../output/streamlit_models/logistic_regression_sentiment_model.pkl")
nb_model = joblib.load("../output/streamlit_models/naive_bayes_sentiment_model.pkl")
vectorizer = joblib.load("../output/streamlit_models/vectorizer.pkl")

# Set Streamlit page configuration
st.set_page_config(page_title="Sentiment Analyzer ", layout="wide", page_icon="💬")

# ---------------------------------- Custom CSS ----------------------------------
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        background-color: #4CAF50; 
        color: white;
        border-radius: 10px;
        padding: 0.6em 2em;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #0E76A8;
        color: white;
        border-radius: 10px;
        padding: 0.6em 2em;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------- Sidebar Theme Toggle ----------------------------------
theme = st.sidebar.radio("🎨 Select Theme", ("Light", "Dark"))
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #111; color: white; }
        .stApp { background-color: #1e1e1e; }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------- App Title ----------------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💬 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover insights in tweets using AI-powered sentiment analysis.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------- Model Selection ----------------------------------
model_choice = st.selectbox("🤖 Choose Sentiment Model", ["Logistic Regression", "Naive Bayes"])
st.markdown(f"✅ You selected: **{model_choice}**")

# ---------------------------------- Tabs ----------------------------------
tab1, tab2 = st.tabs(["📁 Bulk Tweet Analysis", "📝 Single Tweet Analyzer"])

# ------------------------ BULK UPLOAD TAB ------------------------
with tab1:
    st.subheader("📂 Upload Your CSV File")
    upload_file = st.file_uploader("Upload a CSV file of tweets", type="csv")

    if upload_file is not None:
        df = pd.read_csv(upload_file, on_bad_lines="skip")
        st.markdown("### 🔍 Preview of Uploaded Data")
        st.dataframe(df.head())

        st.markdown("### 🔤 Select the Tweet Text Column")
        text_column = st.selectbox("Column containing tweet text", df.columns)

        if st.button("🚀 Run Sentiment Analysis"):
            with st.spinner("Processing... This may take a few seconds"):
                cleaned_texts, predicted_labels = [], []
                progress_bar = st.progress(0)
                total = len(df)

                for i, text in enumerate(df[text_column].astype(str)):
                    cleaned = clean_text(text)
                    model = logreg_model if model_choice == "Logistic Regression" else nb_model
                    pred = model.predict(vectorizer.transform([cleaned]))[0]
                    cleaned_texts.append(cleaned)
                    predicted_labels.append(pred)
                    progress_bar.progress((i + 1) / total)

                df['cleaned_text'] = cleaned_texts
                df['sentiment_code'] = predicted_labels
                df['sentiment'] = df['sentiment_code'].map({
                    "pos": "Positive",
                    "neg": "Negative",
                    "neu": "Neutral"
                })

            st.success("🎉 Sentiment Analysis Complete!")

            sentiment_counts = df['sentiment'].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Sentiment Pie Chart")
                fig1, ax1 = plt.subplots()
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
                ax1.axis('equal')
                st.pyplot(fig1)

            with col2:
                st.markdown("### 📊 Sentiment Bar Chart")
                sns.set(style="whitegrid")
                fig2, ax2 = plt.subplots()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax2, palette="Set2")
                ax2.set_ylabel("Tweet Count")
                st.pyplot(fig2)

            # Word Clouds
            st.markdown("### ☁️ Word Clouds by Sentiment")
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                if words:
                    st.markdown(f"**🗣 {sentiment} Tweets**")
                    wc = WordCloud(width=600, height=300, background_color='white' if theme == "Light" else "black").generate(words)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)

            # Top Tweets
            st.markdown("### 🏆 Top 5 Positive Tweets")
            st.table(df[df['sentiment'] == "Positive"][text_column].head(5))

            st.markdown("### 😡 Top 5 Negative Tweets")
            st.table(df[df['sentiment'] == "Negative"][text_column].head(5))

            # Download button
            csv_result = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv_result, file_name="sentiment_results.csv", mime="text/csv")

# ------------------------ SINGLE TWEET TAB ------------------------
with tab2:
    st.subheader("✏️ Enter a Tweet")
    tweet_input = st.text_area("Type or paste tweet here:", height=100)

    if st.button("🔍 Analyze Tweet"):
        if not tweet_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            cleaned = clean_text(tweet_input)
            model = logreg_model if model_choice == "Logistic Regression" else nb_model
            prediction = model.predict(vectorizer.transform([cleaned]))[0]

            sentiment_map = {
                "pos": "Positive 😊",
                "neg": "Negative 😠",
                "neu": "Neutral 😐"
            }
            st.success(f"**Predicted Sentiment:** `{sentiment_map[prediction]}`")

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.header("📘 About this App")
    st.markdown("""
    This sentiment analyzer helps you classify tweets as **Positive**, **Negative**, or **Neutral** using:
    - Logistic Regression
    - Naive Bayes
    
    Ideal for basic sentiment insights on public opinion from CSV files or individual tweets.
    """)

    st.markdown("---")
    st.caption("📌 Built with ❤️ by Fati")
