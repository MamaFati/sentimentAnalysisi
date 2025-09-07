import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from preprocessing import clean_text
import re
from datetime import datetime

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("⚠️ SpaCy model 'en_core_web_sm' not found. Please install it using: `python -m spacy download en_core_web_sm`")
    st.stop()

# Load models with error handling
try:
    logreg_model = joblib.load("../output/streamlit_models/logistic_regression_sentiment_model.pkl")
    nb_model = joblib.load("../output/streamlit_models/naive_bayes_sentiment_model.pkl")
    vectorizer = joblib.load("../output/streamlit_models/vectorizer.pkl")
except FileNotFoundError:
    st.error("⚠️ Model or vectorizer file not found. Please check the file paths in '../output/streamlit_models/'.")
    st.stop()

# Set Streamlit page configuration
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide", page_icon="💬")

# ---------------------------------- Custom CSS ----------------------------------
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #1e90ff; 
        color: white;
        border-radius: 12px;
        padding: 0.8em 2.5em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: scale(1.05);
    }
    .stDownloadButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 12px;
        padding: 0.8em 2.5em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #e55a50;
        transform: scale(1.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: bold;
        color: #333;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1e90ff;
        border-bottom: 2px solid #1e90ff;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
        font-size: 16px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .sentiment-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        # display: flex;
        margin: 5px;
    }
    .positive { background-color: #4caf50; color: white; }
    .negative { background-color: #ef5350; color: white; }
    .neutral { background-color: #bdbdbd; color: white; }
    .result-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .dark-theme .result-card { background-color: #2a2a2a; color: #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------- Sidebar Theme Toggle ----------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    theme = st.selectbox("🎨 Theme", ["Light", "Dark"], help="Switch between light and dark themes")
    if theme == "Dark":
        st.markdown("""
            <style>
            .main { background-color: #1e1e1e; }
            .stApp { background-color: #1e1e1e; color: white; }
            h1, h2, h3, p { color: #e0e0e0; }
            .stTabs [data-baseweb="tab-list"] { background-color: #2a2a2a; }
            </style>
        """, unsafe_allow_html=True)

# ---------------------------------- App Title ----------------------------------
st.markdown("<h1 style='text-align: center; color: #1e90ff;'>💬 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Unlock insights from tweets with AI-powered sentiment analysis</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------- Model Selection ----------------------------------
st.markdown("<div class='tooltip'>🤖 Choose Model<span class='tooltiptext'>Logistic Regression for balanced accuracy; Naive Bayes for fast text processing</span></div>", unsafe_allow_html=True)
model_choice = st.selectbox("", ["Logistic Regression", "Naive Bayes"], key="model_select")
st.markdown(f"<p style='color: #1e90ff; font-weight: bold;'>✅ Selected: {model_choice}</p>", unsafe_allow_html=True)

# ---------------------------------- Tabs ----------------------------------
tab1, tab2 = st.tabs(["📂 Bulk Tweet Analysis", "✏️ Single Tweet Analysis"])

# ------------------------ BULK UPLOAD TAB ------------------------
with tab1:
    st.subheader("📂 Upload CSV File")
    st.markdown("<p style='color: #666;'>Upload a CSV file containing tweets (e.g., with Tweet_ID, Username, Tweet_Text, etc.).</p>", unsafe_allow_html=True)
    upload_file = st.file_uploader("", type="csv", help="Upload a CSV file with a column containing tweet text.")

    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file, encoding='utf-8', on_bad_lines='skip')
            st.markdown("### 🔍 Data Preview")
            st.markdown(f"**Rows Loaded**: {len(df)}")
            st.dataframe(df.head(), use_container_width=True)

            # Check for missing values in text column
            st.markdown("### 🔤 Select Tweet Text Column")
            text_column = st.selectbox("", df.columns, key="text_column_select", help="Choose the column with tweet text (e.g., Tweet_Text).")

            if text_column not in df.columns:
                st.error("⚠️ Selected column not found in the CSV.")
            else:
                missing_texts = df[text_column].isna().sum()
                if missing_texts > 0:
                    st.warning(f"⚠️ {missing_texts} rows have missing values in '{text_column}'. These will be treated as empty.")

                if st.button("🚀 Analyze Tweets"):
                    with st.spinner("Analyzing tweets... Please wait."):
                        cleaned_texts, predicted_labels, confidence_scores = [], [], []
                        skipped_rows = []
                        progress_bar = st.progress(0)
                        total = len(df)

                        # Batch prediction for efficiency
                        texts = df[text_column].astype(str).tolist()
                        cleaned_texts = [clean_text(text) for text in texts]
                        valid_texts = [text if text.strip() else "" for text in cleaned_texts]
                        
                        # Identify rows to predict (non-empty cleaned texts)
                        predict_indices = [i for i, text in enumerate(valid_texts) if text.strip()]
                        predict_texts = [valid_texts[i] for i in predict_indices]
                        
                        model = logreg_model if model_choice == "Logistic Regression" else nb_model
                        if predict_texts:
                            try:
                                # Vectorize and predict in batch
                                vectorized = vectorizer.transform(predict_texts)
                                predictions = model.predict(vectorized)
                                confidences = [max(proba) * 100 if hasattr(model, "predict_proba") else 0.0 
                                              for proba in (model.predict_proba(vectorized) if hasattr(model, "predict_proba") else [[]])]
                            except Exception as e:
                                st.error(f"⚠️ Batch prediction failed: {str(e)}. Falling back to individual predictions.")
                                predictions, confidences = [], []
                                for i, text in zip(predict_indices, predict_texts):
                                    try:
                                        pred = model.predict(vectorizer.transform([text]))[0]
                                        conf = max(model.predict_proba(vectorizer.transform([text]))[0]) * 100 if hasattr(model, "predict_proba") else 0.0
                                    except Exception as e:
                                        st.warning(f"Prediction failed for row {i+1}: {str(e)}")
                                        pred, conf = "neu", 0.0
                                        skipped_rows.append(i)
                                    predictions.append(pred)
                                    confidences.append(conf)
                        else:
                            predictions, confidences = [], []
                        
                        # Assign predictions to valid texts
                        result_labels = ["neu"] * total
                        result_confs = [0.0] * total
                        for idx, pred, conf in zip(predict_indices, predictions, confidences):
                            result_labels[idx] = pred
                            result_confs[idx] = conf
                        
                        # Log skipped rows (empty after cleaning)
                        for i, text in enumerate(valid_texts):
                            if not text.strip() and i not in skipped_rows:
                                skipped_rows.append(i)
                        
                        df['cleaned_text'] = cleaned_texts
                        df['sentiment_code'] = result_labels
                        df['confidence_score'] = result_confs
                        df['sentiment'] = df['sentiment_code'].map({
                            "pos": "Positive",
                            "neg": "Negative",
                            "neu": "Neutral"
                        }).fillna("Neutral")

                        progress_bar.progress(1.0)

                    st.success(f"🎉 Analysis Complete! Processed {total - len(skipped_rows)}/{total} tweets successfully.")
                    if skipped_rows:
                        st.warning(f"⚠️ Skipped {len(skipped_rows)} rows due to empty text after cleaning or prediction errors. See details below.")

                    # Show skipped rows in an expander
                    if skipped_rows:
                        with st.expander("ℹ️ View Skipped Rows"):
                            skipped_df = df.iloc[skipped_rows][[text_column, 'cleaned_text']]
                            st.dataframe(skipped_df, use_container_width=True)

                    # Summary Card
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown(f"### 📊 Analysis Summary")
                    st.markdown(f"**Total Tweets Analyzed**: {len(df)}")
                    st.markdown(f"**Successfully Processed**: {len(df) - len(skipped_rows)}")
                    st.markdown(f"**Skipped Rows**: {len(skipped_rows)}")
                    sentiment_counts = df['sentiment'].value_counts()
                    for sentiment, count in sentiment_counts.items():
                        badge_class = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}[sentiment]
                        st.markdown(f"<span class='sentiment-badge {badge_class}'>{sentiment}: {count}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Interactive Results Table
                    st.markdown("### 📋 Results Table")
                    st.dataframe(
                        df[[text_column, 'cleaned_text', 'sentiment', 'confidence_score']],
                        use_container_width=True,
                        column_config={
                            text_column: "Original Tweet",
                            "cleaned_text": "Cleaned Tweet",
                            "sentiment": "Sentiment",
                            "confidence_score": st.column_config.NumberColumn(
                                "Confidence (%)", format="%.1f"
                            )
                        }
                    )

                    # Interactive Charts
                    col1, col2 = st.columns([1, 1], gap="medium")
                    with col1:
                        st.markdown("### 📊 Sentiment Distribution")
                        if not sentiment_counts.empty:
                            fig_pie = px.pie(
                                names=sentiment_counts.index,
                                values=sentiment_counts.values,
                                color=sentiment_counts.index,
                                color_discrete_map={"Positive": "#4caf50", "Negative": "#ef5350", "Neutral": "#bdbdbd"},
                                title="Sentiment Distribution"
                            )
                            fig_pie.update_traces(textinfo='percent+label', pull=[0.05] * len(sentiment_counts))
                            fig_pie.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font_color="#333" if theme == "Light" else "#e0e0e0"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.warning("No sentiment data to display.")

                    with col2:
                        st.markdown("### 📈 Sentiment Breakdown")
                        if not sentiment_counts.empty:
                            fig_bar = px.bar(
                                x=sentiment_counts.values,
                                y=sentiment_counts.index,
                                orientation='h',
                                color=sentiment_counts.index,
                                color_discrete_map={"Positive": "#4caf50", "Negative": "#ef5350", "Neutral": "#bdbdbd"},
                                title="Sentiment Breakdown"
                            )
                            fig_bar.update_layout(
                                xaxis_title="Tweet Count",
                                yaxis_title="Sentiment",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font_color="#333" if theme == "Light" else "#e0e0e0",
                                showlegend=False
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.warning("No sentiment data to display.")

                    # Word Clouds in Expander
                    with st.expander("☁️ View Sentiment Word Clouds"):
                        cols = st.columns(3)
                        for idx, sentiment in enumerate(['Positive', 'Negative', 'Neutral']):
                            with cols[idx]:
                                words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'].dropna())
                                if words.strip():
                                    st.markdown(f"**{sentiment}**")
                                    wc = WordCloud(
                                        width=400,
                                        height=200,
                                        background_color='white' if theme == "Light" else "black",
                                        colormap="viridis"
                                    ).generate(words)
                                    fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
                                    ax_wc.imshow(wc, interpolation="bilinear")
                                    ax_wc.axis("off")
                                    st.pyplot(fig_wc)
                                    plt.close(fig_wc)
                                else:
                                    st.markdown(f"**{sentiment}**: No words to display.")

                    # Sample Tweets in Expander
                    with st.expander("🏆 View Sample Tweets by Sentiment"):
                        for sentiment in ['Positive', 'Negative', 'Neutral']:
                            st.markdown(f"**{sentiment} Tweets**")
                            sample = df[df['sentiment'] == sentiment][[text_column, 'confidence_score']].head(3)
                            if not sample.empty:
                                for i, row in enumerate(sample.itertuples(), 1):
                                    st.markdown(f"{i}. {getattr(row, text_column)} (Confidence: {row.confidence_score:.1f}%)")
                            else:
                                st.markdown("No tweets found for this sentiment.")

                    # Download Results
                    csv_result = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Full Results",
                        csv_result,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )

                    # Download Summary Report
                    summary_data = pd.DataFrame({
                        "Metric": ["Total Tweets", "Positive Tweets", "Negative Tweets", "Neutral Tweets", "Skipped Rows"],
                        "Value": [
                            len(df),
                            sentiment_counts.get("Positive", 0),
                            sentiment_counts.get("Negative", 0),
                            sentiment_counts.get("Neutral", 0),
                            len(skipped_rows)
                        ]
                    })
                    summary_csv = summary_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📄 Download Summary Report",
                        summary_csv,
                        file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"⚠️ Error reading CSV file: {str(e)}")

# ------------------------ SINGLE TWEET TAB ------------------------
with tab2:
    st.subheader("✏️ Analyze a Single Tweet")
    st.markdown("<p style='color: #666;'>Enter a tweet to see its sentiment instantly.</p>", unsafe_allow_html=True)
    tweet_input = st.text_area("", placeholder="Type or paste a tweet here...", height=120, key="single_tweet_input")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Analyze Now"):
            if not tweet_input.strip():
                st.warning("⚠️ Please enter a tweet.")
            else:
                with st.spinner("Analyzing..."):
                    cleaned = clean_text(tweet_input)
                    if not cleaned:
                        st.warning("⚠️ Tweet is empty after cleaning.")
                        sentiment, color, emoji = "Neutral", "#bdbdbd", "😐"
                        confidence = 0.0
                    else:
                        try:
                            model = logreg_model if model_choice == "Logistic Regression" else nb_model
                            prediction = model.predict(vectorizer.transform([cleaned]))[0]
                            sentiment_map = {
                                "pos": ("Positive", "#4caf50", "😊"),
                                "neg": ("Negative", "#ef5350", "😣"),
                                "neu": ("Neutral", "#bdbdbd", "😐")
                            }
                            sentiment, color, emoji = sentiment_map.get(prediction, ("Neutral", "#bdbdbd", "😐"))
                            confidence = max(model.predict_proba(vectorizer.transform([cleaned]))[0]) * 100 if hasattr(model, "predict_proba") else 0.0
                        except Exception as e:
                            st.warning(f"Prediction failed: {str(e)}")
                            sentiment, color, emoji, confidence = "Neutral", "#bdbdbd", "😐", 0.0
                    
                    # Store results in session state for display
                    st.session_state.single_tweet_result = {
                        "original": tweet_input,
                        "cleaned": cleaned,
                        "sentiment": sentiment,
                        "color": color,
                        "emoji": emoji,
                        "confidence": confidence
                    }
    
    with col2:
        if st.button("🗑️ Clear Input"):
            st.session_state.single_tweet_input = ""
            if "single_tweet_result" in st.session_state:
                del st.session_state.single_tweet_result
            st.rerun()

    # Display Single Tweet Result
    if "single_tweet_result" in st.session_state:
        result = st.session_state.single_tweet_result
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"### Predicted Sentiment: <span class='sentiment-badge' style='background-color: {result['color']}'>{result['sentiment']} {result['emoji']}</span>")
        if result['confidence'] > 0:
            st.markdown(f"**Confidence**: {result['confidence']:.1f}%")
        st.markdown(f"**Original Tweet**: {result['original']}")
        st.markdown(f"**Cleaned Tweet**: {result['cleaned']}")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This app analyzes tweet sentiment using **Logistic Regression** or **Naive Bayes** models. Upload a CSV (e.g., with Tweet_Text column) for bulk analysis or test a single tweet. Perfect for understanding public opinion on X!
    """)
    st.markdown("---")
    st.caption("📌 Built with ❤️ by Fati | Powered by Streamlit")