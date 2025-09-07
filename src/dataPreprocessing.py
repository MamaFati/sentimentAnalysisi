# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.naive_bayes import MultinomialNB
from preprocessing import clean_text
import joblib
warnings.filterwarnings("ignore")
from wordcloud import WordCloud
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from textblob import TextBlob, Word, Blobber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, roc_curve
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import seaborn as sns
import os
from sklearn.preprocessing import label_binarize
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")
nltk.download("punkt_tab")
sid = SentimentIntensityAnalyzer()
from sklearn.metrics import accuracy_score
# import transformers
# print(transformers.__file__)
# print(transformers.__version__)



# === Ensure output directories exist ===
os.makedirs("../images", exist_ok=True)
os.makedirs("../output/streamlit_models", exist_ok=True)


# Define label_map
label_map = {"pos": 0, "neu": 1, "neg": 2}

# === Utilities ===
stemmer = PorterStemmer() 
def char_counts(x):
  s = x.split()
  x = "".join(s)
  return len(x)

def stemming(data):
    words = data.split()   
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

def polarity(text):
    return TextBlob(text).sentiment.polarity

def get_top_words(df, label, n=10):
    words = " ".join(df[df["sentiment"] == label]["text"]).split()
    common = Counter(words).most_common(n)
    return pd.DataFrame(common, columns=["Word", "Frequency"])

# #  Adding Sentiment to the data frame
def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

###---- DATA COLLECTION ---###
# load dataset
df = pd.read_csv("../ data/rawData/vaccination_all_tweets.csv")
df.info()
df.shape   
 
 #### LOAD AND PREPROCESSING DATA ####
reduceData = df[:30000]
 
# Drop all columns excluding the text
content_df =  reduceData.drop(["id", "user_name", "user_location", "user_description", "user_created",
       "user_followers", "user_friends", "user_favourites", "user_verified",
       "date", "hashtags", "source", "retweets", "favorites",
       "is_retweet"],axis=1)
    
content_df["text"] = content_df["text"].apply(clean_text)
content_df["word_counts"] = content_df["text"].apply(lambda x: len(x.split()))
content_df["char_counts"] = content_df["text"].apply(char_counts)
content_df["avg_word_len"] = content_df["char_counts"] / content_df["word_counts"]
content_df["stop_words_len"] = content_df["text"].apply(lambda x: len([t for t in x.split() if t in stopwords]))
content_df["twitts_no_stop"] = content_df["text"].apply(lambda x: " ".join([t for t in x.split() if t not in stopwords]))
content_df["hastags_count"] = content_df["text"].apply(lambda x: len([t for t in x.split() if t.startswith("#")]))
content_df["hastags_mention"] = content_df["text"].apply(lambda x: len([t for t in x.split() if t.startswith("@")]))
content_df["numerics_count"] = content_df["text"].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
content_df["emails"] = content_df["text"].apply(lambda x: re.findall(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", x))
content_df["emails_count"] = content_df["emails"].apply(len)


### ---- Word Cloud Visualization --- ###
# Combine all text into one big string
text = " ".join(content_df["text"].dropna().tolist())
# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Tweets", fontsize=20)
plt.savefig("../output/images/visualization/generalWordCloud.png")
plt.close()
content_df.to_csv("cleaned_tweets.csv", index=False)
# print(content_df["text"])



###--- Data Analysis and Visualization ---###
# # Lists to store words based on sentiment
text
# print(text)
negative_words = []
positive_words = []
neutral_words = []

words = text.split()
# # Perform sentiment analysis for each word
for word in words:
    blob = TextBlob(word)
    # blob.sentiment.polarity is a feature provided by TextBlob that quantifies the sentiment of a text.
    sentiment_score = blob.sentiment.polarity
    words = blob.words
    if sentiment_score < 0:
        negative_words.extend(words)
    elif sentiment_score > 0:
        positive_words.extend(words)
    else:
        neutral_words.extend(words)

 
if negative_words:
    negative_wordcloud = WordCloud(width=800, height=400, background_color="white")
    negative_wordcloud.generate_from_frequencies(dict(zip(negative_words, [1] * len(negative_words))))
    plt.figure(figsize=(6, 4))
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.title("Negative Sentiment")
    plt.axis("off")
    plt.savefig("../output/images/visualization/negativeWordCloud.png")
    plt.close()
else:
    print("No negative words available for word cloud.")

# Generate word cloud for positive words if available
if positive_words:
    positive_wordcloud = WordCloud(width=800, height=400, background_color="white")
    positive_wordcloud.generate_from_frequencies(dict(zip(positive_words, [1] * len(positive_words))))
    plt.figure(figsize=(6, 4))
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.title("Positive Sentiment")
    plt.axis("off")
    plt.savefig("../output/images/visualization/positiveWordCloud.png")
    plt.close()
else:
    print("No positive words available for word cloud.")

# Generate word cloud for neutral words if available
if neutral_words:
    neutral_wordcloud = WordCloud(width=800, height=400, background_color="white")
    neutral_wordcloud.generate_from_frequencies(dict(zip(neutral_words, [1] * len(neutral_words))))
    plt.figure(figsize=(6, 4))
    plt.imshow(neutral_wordcloud, interpolation="bilinear")
    plt.title("Neutral Sentiment")
    plt.axis("off")
    plt.savefig("../output/images/visualization/neutralWordCloud.png")
    plt.close()
else:
    print("No neutral words available for word cloud.")



##### Analysis individual Tweet Sentiment analysis

sentiment_df = pd.DataFrame()
sentiment_df["text"] = content_df["text"].copy()
# print("Shape of data after processing:",sentiment_df["text"].shape)  

# ## Performing Porter stemming in NLP is done to reduce words to their base or root form, known as stems. This process helps in simplifying text data and improving text analysis tasks like information retrieval, sentiment analysis, and topic modeling. By reducing words to their stems, variations of a word are collapsed into a single representation, which can enhance the efficiency and effectiveness of NLP algorithms.
stemmer = PorterStemmer()
# apply stemming to the text 
sentiment_df["text"] = sentiment_df["text"].apply(lambda x: stemming(x))
sentiment_df["scores"] = sentiment_df["text"].apply(lambda review: sid.polarity_scores(review))
# # This code calculates sentiment scores for each text in the "text" column of a DataFrame called sentiment_df.
sentiment_df["compound"]  = sentiment_df["scores"].apply(lambda score_dict: score_dict["compound"])
sentiment_df["comp_score"] = sentiment_df["compound"].apply(lambda c: "pos" if c > 0 else ("neg" if c < 0 else "neu"))
# #calculating polarity for categorizing text
sentiment_df["polarity"] = sentiment_df["text"].apply(polarity)
sentiment_df_list = sentiment_df.copy()
sentiment_df_list["sentiment"] = sentiment_df_list["polarity"].apply(sentiment)

 ### --- Visualization ---###
fig = plt.figure(figsize=(7,5))
sns.countplot(x="sentiment",data=sentiment_df_list)
fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {"linewidth":2, "edgecolor":"black"}
tags = sentiment_df_list["sentiment"].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind="pie", autopct="%1.1f%%", shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label="")
plt.title("Distribution of sentiments")


# Visulaizing Top 5 positive Sentiments
pos_tweets = sentiment_df_list[sentiment_df_list.sentiment == "Positive"]
pos_tweets = pos_tweets.sort_values(["polarity"], ascending= False)
pos_tweets.head()

# Visualizing the Sentiment
fig = plt.figure(figsize=(7,5))
sns.countplot(x="comp_score",data=sentiment_df_list)

fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {"linewidth":2, "edgecolor":"black"}
tags = sentiment_df_list["comp_score"].value_counts()
explode = tuple([0.1] * len(tags))   

colors = ("yellowgreen", "gold", "red")[:len(tags)]   
wp = {"linewidth": 2, "edgecolor": "black"}

tags.plot(
    kind="pie",
    autopct="%1.1f%%",
    shadow=True,
    colors=colors,
    startangle=90,
    wedgeprops=wp,
    explode=explode,
    label=""
)
plt.title("Distribution of sentiments comp_score")
plt.tight_layout()
plt.savefig("../output/images/visualization/sentimentComp_score.png")
plt.close("all")

df["date"] = pd.to_datetime(df["date"])  
sentiment_df_list["date"] = df["date"][:len(sentiment_df_list)]

daily_sentiment = sentiment_df_list.groupby("date")["compound"].mean()

plt.figure(figsize=(12, 6))
daily_sentiment.plot()
plt.title("Average Compound Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Average Compound Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/images/visualization/CompoundScore.png")
plt.close("all")
# #For  all numeric columns  
correlation_data = content_df[["word_counts", "char_counts", "avg_word_len", "stop_words_len", "hastags_count", "hastags_mention", "numerics_count" ]]
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Tweet Text Features")
plt.tight_layout()
plt.savefig("../output/images/visualization/correlationOfTweetFeatures.png")
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# #First subplot for compound scores
sns.countplot(x="comp_score", data=sentiment_df_list, ax=axes[0])
axes[0].set_title("Compound Score Sentiment")

##Second subplot for sentiments
sns.countplot(x="sentiment", data=sentiment_df_list, ax=axes[1])
axes[1].set_title("Sentiment Labels")
plt.tight_layout()
plt.savefig("../output/images/visualization/sentiment-label.png")
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# First subplot for compound score distribution
colors = ("yellowgreen", "gold", "red")
wp = {"linewidth": 2, "edgecolor": "black"}
tags_comp = sentiment_df_list["comp_score"].value_counts()
explode = tuple([0.1] * len(tags_comp))
colors = ("yellowgreen", "gold", "red")[:len(tags_comp)]  # Also dynamic color support

tags_comp.plot(
    kind="pie",
    autopct="%1.1f%%",
    shadow=True,
    colors=colors,
    startangle=90,
    wedgeprops=wp,
    explode=explode,
    label="",
    ax=axes[0]
)
axes[0].set_title("Distribution of Compound Score Sentiments")

# Second subplot for sentiment label distribution
tags_sentiment = sentiment_df_list["sentiment"].value_counts()
explode = tuple([0.1] * len(tags_sentiment))   
colors = sns.color_palette("Set2", len(tags_sentiment)) 
 
tags_sentiment.plot(
    kind="pie",
    autopct="%1.1f%%",
    shadow=True,
    colors=colors,
    startangle=90,
    wedgeprops=wp,
    explode=explode,
    label="",
    ax=axes[1]
)
axes[1].set_title("Distribution of Sentiment Labels")

plt.title("Distribution of sentiments", fontsize=14)
plt.legend(title="Sentiment")
plt.ylabel("")
plt.tight_layout()
plt.savefig("../output/images/visualization/sentiment_distribution.png")
plt.close("all")
# #####----- Model Evaluation ---- #####
 
vect = CountVectorizer(ngram_range=(1, 2)).fit(sentiment_df["text"])
X = vect.transform(sentiment_df["text"])
Y = sentiment_df["comp_score"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistic Regression Model
logreg = LogisticRegression(class_weight="balanced")
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
print("LogReg Sentiment Accuracy: {:.2f}%".format(accuracy_score(y_test, logreg_pred)*100))
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
print("Naive Bayes Sentiment Accuracy: {:.2f}%".format(accuracy_score(y_test, nb_pred)*100))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Save Models
joblib.dump(vect, "vectorizer.pkl")
joblib.dump(logreg, "sentiment_model.pkl")
joblib.dump(nb_model, "naive_bayes_sentiment_model.pkl")

# === Save Results to TXT ===
with open("../output/sentiment_model_final_comparison_report.txt", "w") as f:
    f.write("===== Logistic Regression Model =====\n")
    f.write(f"Accuracy: {accuracy_score(y_test, logreg_pred) * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, logreg_pred)) + "\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, logreg_pred, target_names=list(label_map.keys())))
    f.write("\n\n")

    f.write("===== Naive Bayes Model =====\n")
    f.write(f"Accuracy: {accuracy_score(y_test, nb_pred) * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, nb_pred)) + "\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, nb_pred, target_names=list(label_map.keys())))
    f.write("\n\n")

# === Comparative Visualizations ===
 
 # Define class labels
labels = ["pos", "neu", "neg"]

# Get precision, recall, and f1-score for both models
precision_log, recall_log, f1_log, _ = precision_recall_fscore_support(
    y_test, logreg_pred, labels=labels, zero_division=0
)
precision_nb, recall_nb, f1_nb, _ = precision_recall_fscore_support(
    y_test, nb_pred, labels=labels, zero_division=0
)

# Create DataFrame for visualization
metrics_df = pd.DataFrame({
    "Class": labels * 2,
    "Precision": list(precision_log) + list(precision_nb),
    "Recall": list(recall_log) + list(recall_nb),
    "F1-Score": list(f1_log) + list(f1_nb),
    "Model": ["Logistic Regression"] * len(labels) + ["Naive Bayes"] * len(labels)
})

# -------- Precision Plot --------
plt.figure(figsize=(8, 5))
sns.barplot(x="Class", y="Precision", hue="Model", data=metrics_df, palette="Set2")
plt.title("Precision Comparison per Class")
plt.ylim(0, 1)
plt.ylabel("Precision")
plt.savefig("../output/images/resultAnalysisVisualization/precision_comparison.png")
plt.show()

# -------- Recall Plot --------
plt.figure(figsize=(8, 5))
sns.barplot(x="Class", y="Recall", hue="Model", data=metrics_df, palette="Set1")
plt.title("Recall Comparison per Class")
plt.ylim(0, 1)
plt.ylabel("Recall")
plt.savefig("../output/images/resultAnalysisVisualization/recall_comparison.png")
plt.show()

# -------- F1-Score Plot --------
plt.figure(figsize=(8, 5))
sns.barplot(x="Class", y="F1-Score", hue="Model", data=metrics_df, palette="Set3")
plt.title("F1-Score Comparison per Class")
plt.ylim(0, 1)
plt.ylabel("F1-Score")
plt.savefig("../output/images/resultAnalysisVisualization/f1score_comparison.png")
plt.show()

# --- Combined Metrics Visualization ---

# Reshape metrics for grouped bar chart
combined_df = pd.DataFrame({
    "Class": labels * 6,
    "Metric": (["Precision"] * len(labels) + ["Recall"] * len(labels) + ["F1-Score"] * len(labels)) * 2,
    "Score": list(precision_log) + list(recall_log) + list(f1_log) +
             list(precision_nb) + list(recall_nb) + list(f1_nb),
    "Model": ["Logistic Regression"] * (3*len(labels)) + ["Naive Bayes"] * (3*len(labels))
})

plt.figure(figsize=(12, 6))
sns.barplot(x="Class", y="Score", hue="Metric", data=combined_df, palette="Set2", ci=None)

# Add grid and labels
plt.title("Comparison of Precision, Recall, and F1-Score per Class")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.legend(title="Metric")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save figure
plt.tight_layout()
plt.savefig("../output/images/resultAnalysisVisualization/combined_metrics_comparison.png")
plt.show()


# Accuracy Comparison Bar Chart
model_names = ["Logistic Regression", "Naive Bayes"]
accuracy_scores_list = [
    accuracy_score(y_test, logreg_pred) * 100,
    accuracy_score(y_test, nb_pred) * 100,
]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracy_scores_list, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig("../output/images/resultAnalysisVisualization/model_accuracy_comparison.png")
plt.close()

# Confusion Matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, logreg_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("../output/images/resultAnalysisVisualization/confusion_matrix_logreg.png")
plt.close()

# Confusion Matrix for Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, nb_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("../output/images/resultAnalysisVisualization/confusion_matrix_nb.png")
plt.close()

# Convert labels to binary format for multi-class ROC (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=["pos", "neu", "neg"])
n_classes = y_test_bin.shape[1]

# Predict probabilities for ROC
logreg_probs = logreg.predict_proba(x_test)
nb_probs = nb_model.predict_proba(x_test)

# Plot ROC Curve for each class
fpr_logreg = dict()
tpr_logreg = dict()
roc_auc_logreg = dict()

fpr_nb = dict()
tpr_nb = dict()
roc_auc_nb = dict()

for i in range(n_classes):
    fpr_logreg[i], tpr_logreg[i], _ = roc_curve(y_test_bin[:, i], logreg_probs[:, i])
    roc_auc_logreg[i] = auc(fpr_logreg[i], tpr_logreg[i])

    fpr_nb[i], tpr_nb[i], _ = roc_curve(y_test_bin[:, i], nb_probs[:, i])
    roc_auc_nb[i] = auc(fpr_nb[i], tpr_nb[i])

# Plotting the ROC curves for Logistic Regression
plt.figure(figsize=(10, 6))
colors = ["red", "green", "blue"]
for i, label in enumerate(label_map.keys()):
    plt.plot(fpr_logreg[i], tpr_logreg[i], color=colors[i], lw=2,
             label=f"LogReg ROC curve (class {label}) - AUC = {roc_auc_logreg[i]:.2f}")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../output/images/resultAnalysisVisualization/roc_curve_logreg.png")
plt.close()

# Plotting the ROC curves for Naive Bayes
plt.figure(figsize=(10, 6))
for i, label in enumerate(label_map.keys()):
    plt.plot(fpr_nb[i], tpr_nb[i], color=colors[i], lw=2,
             label=f"Naive Bayes ROC curve (class {label}) - AUC = {roc_auc_nb[i]:.2f}")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Bayes")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../output/images/resultAnalysisVisualization/roc_curve_naive_bayes.png")
plt.close()

# ConfusionMatrixDisplay for Logistic Regression
disp_logreg = ConfusionMatrixDisplay.from_estimator(
    logreg, x_test, y_test, display_labels=list(label_map.keys()), cmap="Blues", normalize=None
)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("../output/images/resultAnalysisVisualization/confusion_display_logreg.png")
plt.close()

# ConfusionMatrixDisplay for Naive Bayes
disp_nb = ConfusionMatrixDisplay.from_estimator(
    nb_model, x_test, y_test, display_labels=list(label_map.keys()), cmap="Blues", normalize=None
)
plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.savefig("../output/images/resultAnalysisVisualization/confusion_display_naive_bayes.png")
plt.close()

print("ROC curves and additional confusion matrices saved.")


# Save CountVectorizer and Models for Streamlit
joblib.dump(vect, "../output/streamlit_models/vectorizer.pkl")
joblib.dump(logreg, "../output/streamlit_models/logistic_regression_sentiment_model.pkl")
joblib.dump(nb_model, "../output/streamlit_models/naive_bayes_sentiment_model.pkl")

print("Model training and evaluation completed successfully!")
print("Results saved to sentiment_model_final_comparison_report.txt")
print("Visualizations saved to output/images/")
print("Models saved to output/streamlit_models/")

 

