from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl") 

app = FastAPI()

@app.get("/predict")
def predict_sentiment(text: str):
    # Convert the input text to feature vector
    X_new = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(X_new)
    
    return {"text": text, "predicted_sentiment": prediction[0]} 



