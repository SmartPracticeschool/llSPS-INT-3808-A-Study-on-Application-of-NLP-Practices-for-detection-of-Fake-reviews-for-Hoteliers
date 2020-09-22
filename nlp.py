# -*- coding: utf-8 -*-

import pandas as pd
dataset=pd.read_csv(r'data\deceptive-opinion.csv')
import re
import nltk #natural language tool kit
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,1600):
    
    review=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    c.append(review)

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(c).toarray()
joblib.dump(cv.vocabulary_,"features.save")
y=dataset.iloc[:,0].values
lb=LabelEncoder()
y=lb.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(input_dim=2000,units=20,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dense(units=10,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=80)

y_pred=model.predict(x_test)
y_pred=(y_pred>=0.5)

loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load('features.save'))

da="My husband and I went to Chicago for three nights for a quick get-away. We chose Fairmont Chicago Millennium Park because it was located at the heart of the city, downtown Chicago. The hotel itself was very elegant and modern. The bed was extremely comfortable and I was able to wake up with music from my iPod. I would definitely stay with them again."
da=da.split('delimeter')
result=model.predict(loaded.transform(da))
prediction=result>=0.5
print(prediction)

model.save('model1.h5')