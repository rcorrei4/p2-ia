import pandas as pd 
import numpy as np     
from nltk.corpus import stopwords  
from sklearn.model_selection import train_test_split       
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential    
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.callbacks import ModelCheckpoint  
from tensorflow.keras.models import load_model 
import re

data = pd.read_csv('IMDB Dataset.csv')

print(data)
english_stops = set(stopwords.words('english'))
def load_dataset():
    df = pd.read_csv('IMDB Dataset.csv')
    x_data = df['review']   
    y_data = df['sentiment']    

    x_data = x_data.replace({'<.*?>': ''}, regex = True)    
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)  
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops]) 
    x_data = x_data.apply(lambda review: [w.lower() for w in review]) 
    
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data

x_data, y_data = load_dataset()

print('Reviews')
print(x_data, '\n')
print('Sentiment')
print(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

print('Train Set')
print(x_train, '\n')
print(x_test, '\n')
print('Test Set')
print(y_train, '\n')
print(y_test)

def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))

token = Tokenizer(lower=False)
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1

print('Encoded X Train\n', x_train, '\n')
print('Encoded X Test\n', x_test, '\n')
print('Maximum review length: ', max_length)
EMBED_DIM = 32
LSTM_OUT = 64

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length = max_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model.summary())
checkpoint = ModelCheckpoint(
    'models/LSTM.keras',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)
model.fit(x_train, y_train, batch_size = 128, epochs = 5, callbacks=[checkpoint])

y_pred = np.argmax(model.predict(x_test), axis=-1)

true = 0
for i, y in enumerate(y_test):
    if y == y_pred[i]:
        true += 1

print('Correct Prediction: {}'.format(true))
print('Wrong Prediction: {}'.format(len(y_pred) - true))
print('Accuracy: {}'.format(true/len(y_pred)*100))