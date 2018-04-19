from flask import Flask, render_template, request
import pandas as pd
import json 
import keras.models
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
import keras
from keras import backend as K
from flask import Flask, request, redirect, jsonify, render_template

app = Flask(__name__)
model = None
graph = None

PATH = './Sentiment_Model/optimal_dict3.json'
with open(PATH) as json_data:
    bag_of_words = json.load(json_data)
word_dict = pd.Series(bag_of_words)

# Parameters
version = 4
words = len(word_dict)
review_len = 1000
vec_len = 300
patience = 5
batch_size = 40
epochs = 3

#Load the model
def load_model():
    global model
    global graph
    model = keras.models.load_model("./Sentiment_Model/optimalfloyds3.h5")
    graph = K.get_session().graph

load_model()

#Econding functions
def encode_sentence(text):
    result = []
    arr = text_to_word_sequence(text, lower=True, split=" ")
    for word in arr:
        w = encode_word(word)
        if w is not None:
            result.append(w)
    return result

def encode_word(word):
    if word not in word_dict:
        return None
    return word_dict[word]

def encode_batch(arr):
    result = []
    for sentence in arr:
        result.append(encode_sentence(sentence))
    return sequence.pad_sequences(result, maxlen=review_len)

def predict_batch(arr):
    batch = encode_batch(arr)
    result = model.predict(batch, batch_size=len(batch), verbose=0)
    return result

#Run Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index.html')
def indexNavigation():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/sniffer.html")
def sniffer():
    return render_template("/sniffer.html")

@app.route('/prediction', methods=['GET', 'POST'],)
def predict():
    data = {"success": False}
    if request.method == 'POST':
        review =  request.form['userInput']
        # Get the tensorflow default graph
        global graph
        with graph.as_default():

            # Use the model to make a prediction
            prediction = predict_batch([review])
            
            # Format the data in a useable form
            data["prediction_string"] = str(prediction)
            output = data['prediction_string'][2:7]
            output2 = float(output)
    return jsonify(output2)

if __name__ == "__main__":
    load_model()
    app.run()