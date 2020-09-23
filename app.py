# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
app = Flask(__name__)
from tensorflow import keras
model = keras.models.load_model('model1.h5')

@app.route('/')
def home():
    return render_template('homepage.html')
    
@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    da = [[x for x in request.form.values()]]
    da = da[0][0]
    print(da)
    
    loaded=CountVectorizer(decode_error='replace',vocabulary=load('features.save'))
    
    da=da.split('delimeter')
    result=model.predict(loaded.transform(da))
    print(result)
    if result>=0.5:
        output="Real"
    else:
        output="Fake"
        
    
    return render_template('index.html', prediction_text='This is a {} review'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
