from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load("disaster_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    tweet = ""

    if request.method == "POST":

        tweet = request.form["tweet"]

        # Clean the user's tweet
        cleaned_tweet = clean_text(tweet)

        # Convert tweet into TF-IDF features
        tweet_vector = tfidf.transform([cleaned_tweet])

        # Make prediction
        prediction_result = model.predict(tweet_vector)[0]

        # Get disaster probability
        probability = model.predict_proba(tweet_vector)[0][1] * 100

        if prediction_result == 1:
            prediction = "Disaster Tweet"
        else:
            prediction = "Not a Disaster Tweet"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        tweet=tweet
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)