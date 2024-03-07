from pycaret import classification
import pickle
from flask import Flask, request, jsonify, make_response
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import pandas as pd

stopwords_set = set(stopwords.words("english"))


def preprocess(s: str):
    ps = PorterStemmer()
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"[^A-Za-z]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = word_tokenize(s)
    s = [word for word in s if not word in stopwords_set]
    s = [ps.stem(w) for w in s]
    s = " ".join(s).strip()
    return s


app = Flask(__name__)
app.tfidf_vectorizer = pickle.load(open("resources/text_vectorizer.pkl", "rb"))
app.columns = pickle.load(open("resources/columns.pkl", "rb"))
app.model = classification.load_model("resources/fake_news_pipeline")


@app.route("/predict", methods=["GET"])
def predict_basic():
    response_object = {"status": "success"}
    argList = request.args.to_dict(flat=False)
    title = argList["title"][0]
    body = argList["body"][0]
    predict = classification.predict_model(
        app.model,
        pd.DataFrame.sparse.from_spmatrix(
            app.tfidf_vectorizer.transform([" ".join([title, body])]),
            columns=app.columns[:-1],
        ),
    ).loc[0]
    response_object["predict_as"] = (
        "Fake" if predict["prediction_label"] > 0.5 else "True"
    )
    response_object["probability"] = predict["prediction_score"]
    return make_response(
        jsonify(response_object), 200, {"Access-Control-Allow-Origin": "*"}
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
