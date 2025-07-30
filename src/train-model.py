# Performing Vectorizing to create bigram model
vect = CountVectorizer(ngram_range=(1,2)).fit(sentiment_df_list['text'])

#seperating Independent and Depentent Variables and tranform X data
X = sentiment_df_list['text'] # for text
Y = sentiment_df_list['sentiment'] # for sentiment text
X = vect.transform(X)

# Splitting data with test 20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#Checking shape of train and test data
# print("Size of x_train:", (x_train.shape))
# print("Size of y_train:", (y_train.shape))
# print("Size of x_test:", (x_test.shape))
# print("Size of y_test:", (y_test.shape))


#Training logisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

#Confusion matrix
print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))

# Model Development comp_score which find with SentimentIntensityAnalyzer
#seperating Independent and Depentent Variables and tranform X data
X = sentiment_df_list['text']
Y = sentiment_df_list['comp_score']
X = vect.transform(X)

# Splitting data with test 20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Training logisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

#Training logisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

# Save vectorizer and model
joblib.dump(vect, 'vectorizer.pkl')
joblib.dump(logreg, 'sentiment_model.pkl')