import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(review):
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = new_model.predict(padded_sequence)
  sentiment = "positiva" if prediction[0][0] > 0.5 else "negativa"
  return sentiment

# Carregar model e tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
new_model = load_model('LSTM.keras')

new_review = str(input('Movie Review: '))
sentiment = predict_sentiment(new_review)
print(f"Essa avaliação foi: {sentiment}")