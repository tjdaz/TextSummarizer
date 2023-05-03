import numpy as np
import tensorflow as tf
import helper_functions as hf
from keras.utils import pad_sequences
import pickle

# Load the saved model
model = tf.keras.models.load_model('data/models/glove_new_model_01.h5')

# Load the tokenizer and max sequence lengths from disk
with open('data/models/glove_new_model_tokenizer_01.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set max lengths.
max_article_length = 1516
max_summary_length = 70

# Load the article from the input file
doc = hf.get_doc_by_id(3000)
article_text = doc[2]
summary_text = doc[3]

# Tokenize the article and add start/end tokens
article_seq = tokenizer.texts_to_sequences(['<start> ' + article_text + ' <end>'])

# Pad the sequence to the max length
article_input = pad_sequences(article_seq, maxlen=max_article_length)

# Generate the summary
predicted_summary = model.predict([article_input, np.zeros((1, max_summary_length))])[0]
predicted_summary = np.argmax(predicted_summary, axis=1)  # convert one-hot encoding to integer indices
predicted_summary = ' '.join(tokenizer.index_word[i] for i in predicted_summary if i > 0)  # filter out padding and OOV tokens
predicted_summary = predicted_summary.replace('<start> ', '').replace(' <end>', '')

# Print the summary
print('Actual:\n' + summary_text)
print('\nPredicted:\n' + predicted_summary)
