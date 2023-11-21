from flask import *
from flask import render_template
app = Flask(__name__)
import numpy as np
import pandas as pd
import nltk
import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet                                               
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup
wordnetlemmatizer= WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
def lemmatize_sentence(sentence):
    wordnetlemmatizer= WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    lemmatized_words = [wordnetlemmatizer.lemmatize(i,pos= 'n') for i in tokens]
    return ' '.join(lemmatized_words)
def remove_stopwords(tokens):
  tokens_list = []
  stop_words = set(stopwords.words('english'))
  for word in tokens:
      if word.lower() not in stop_words:
          tokens_list.append(word.lower())
  return list(filter(lambda x: len(x) > 1, tokens_list))
def classify_web(url_input,clf):
    tokens=[]
    try:
        res = requests.get(url_input)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text()
            cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
            tokens = word_tokenize(cleaned_text)
            tokens=remove_stopwords(tokens)
            lemmatizated_words = [wordnetlemmatizer.lemmatize(i,pos= 'n') for i in tokens]
        else:
            print(f'Request failed ({res.status_code}). Please check if website do not blocking or it is still existing')
    except Exception as e:
        print(f'Request to {url_input} failed. Error code:\n {e}')
    text_from_html = ' '.join(lemmatizated_words)
    text_from_html = text_from_html.lower()
    # Create a vector for the input website
    X_input = vectorizer.transform([text_from_html])
    # Classify the input website using the trained classifier
    prediction = clf.predict(X_input)[0]
    return prediction
url = "https://raw.githubusercontent.com/20wh1a0519/mlProject/main/updated_website_data.csv"
df = pd.read_csv(url)
df['cleaned_website_text'] = df['cleaned_website_text'].apply(lambda x: lemmatize_sentence(x))
df = df[df.category != 'Adult']
df = df[df.category != 'Forums']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_website_text"], df["category"], test_size=0.2, random_state= 35)

# Create a vector using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)

# Train the Logistic Regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)
@app.route('/')
def home_page():
   return render_template('home.html')

@app.route('/index')
def index_page():
   return  render_template('index.html',category = "")

@app.route("/classify", methods= ["POST"])
def classify():
    if request.method == "POST":
        url_input = str(request.form["url"])
        x = "Category : " +  classify_web(url_input,clf)
        print(x)
        return  render_template('index.html',category = x)

if __name__ == '__main__':
   app.run()