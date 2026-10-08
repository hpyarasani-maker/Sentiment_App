from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

#simple training data
texts = [
    "I love this product",
    "this is terrible",
    "Amazing experience",
    "Worst ever",
    "I am very happy with this",
    "I hate it",
    "bad movie",
    "great service"]


labels = ["positive", "negative", "positive", "negative", "positive", "negative", "negative", "positive"]

#convert text data to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

#train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X, labels)

# Save the trained model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
# vector_numbers = X.toarray()
# for text, vector, label in zip(texts, vector_numbers, labels):
#     print(f"Text:   '{text}'")
#     print(f"Vector: {vector}")
#     print(f"Label:  {label}")
#     print("-" * 50)
print("Model and vectorizer saved successfully.")
