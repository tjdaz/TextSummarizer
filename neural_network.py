import os
import sqlite3
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle

# Load the data from SQLite3 database
print('loading data...')
conn = sqlite3.connect('data/database/documents.db')
cur = conn.cursor()
cur.execute("SELECT doc_text, doc_summary FROM documents LIMIT 50")
articles, summaries = zip(*cur.fetchall())
conn.close()
print('loading data complete')

# Tokenize the text data and add <start> and <end> tokens
print('tokenizing text...')
tokenizer = Tokenizer(filters='', oov_token='<OOV>', lower=True)
tokenizer.fit_on_texts(['<start> ' + article + ' <end>' for article in articles] + ['<start> ' + summary + ' <end>' for summary in summaries])
article_sequences = tokenizer.texts_to_sequences(['<start> ' + article + ' <end>' for article in articles])
summary_sequences = tokenizer.texts_to_sequences(['<start> ' + summary + ' <end>' for summary in summaries])
print('tokenizing text complete')

# Pad the sequences to a fixed length
print('padding sequences...')
max_article_length = max(len(seq) for seq in article_sequences)
max_summary_length = max(len(seq) for seq in summary_sequences)
article_input = pad_sequences(article_sequences, maxlen=max_article_length)
summary_input = pad_sequences(summary_sequences, maxlen=max_summary_length)
print('padding sequences complete')

# Load pre-trained GloVe embeddings
embedding_dim = 300
embedding_file = 'data/glove/glove.6B.300d.txt'
embedding_index_file = 'data/glove/glove.6B.300d.txt.pkl'

print('creating embedding index file...')
if os.path.isfile(embedding_index_file):
    with open(embedding_index_file, 'rb') as f:
        embedding_index = pickle.load(f)
else:
    embedding_index = {}
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    with open(embedding_index_file, 'wb') as f:
        pickle.dump(embedding_index, f)
print('embedding index file created')


# Create embedding matrix
print('creating embedding matrix...')
word_index = tokenizer.word_index
num_words = min(len(word_index) + 1, len(embedding_index))
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('embedding matrix created')

# Build the model
print('building model...')
article_input_layer = tf.keras.layers.Input(shape=(max_article_length,))
summary_input_layer = tf.keras.layers.Input(shape=(max_summary_length,))
embedding_layer = tf.keras.layers.Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=True)
article_embedding = embedding_layer(article_input_layer)
summary_embedding = embedding_layer(summary_input_layer)
article_lstm_layer = tf.keras.layers.LSTM(100)(article_embedding)
summary_lstm_layer = tf.keras.layers.LSTM(100)(summary_embedding)
merged_layer = tf.keras.layers.concatenate([article_lstm_layer, summary_lstm_layer])

# set to linear
output_layer = tf.keras.layers.Dense(max_summary_length * num_words, activation='tanh')(merged_layer)
output_layer = tf.keras.layers.Reshape((max_summary_length, num_words))(output_layer)
model = tf.keras.models.Model(inputs=[article_input_layer, summary_input_layer], outputs=output_layer)

# Compile the model
model.compile(loss='cosine_similarity', optimizer='adam', metrics=['accuracy'])
print('model built/compiled')

# Train the model
print('training model...')
epochs = 10
batch_size = 2
for epoch in range(epochs):
    for i in range(0, len(article_input), batch_size):
        # Convert the summary input into one-hot encoded vectors
        summary_one_hot = tf.one_hot(summary_input[i:i + batch_size], num_words)

        # Train the model on batches of article and summary inputs and their one-hot encoded vectors
        loss, acc = model.train_on_batch(
            [article_input[i:i + batch_size], summary_input[i:i + batch_size]],
            summary_one_hot)

    # Predict the summary for a single article. this should just print out a summary and not affect training.
    index = 0  # index of the article you want to generate a summary for
    article = article_input[index]
    article = np.expand_dims(article, axis=0)  # add batch dimension
    predicted_summary = model.predict([article, summary_input[index:index+1]])[0]
    predicted_summary = np.argmax(predicted_summary, axis=1)  # convert one-hot encoding to integer indices
    predicted_summary = ' '.join(tokenizer.index_word[i] for i in predicted_summary if i > 0)  # convert integer indices to words
    print('Epoch {}: loss = {}, acc = {}'.format(epoch+1, loss, acc))
    print('Epoch {}: predicted summary for article {}: {}'.format(epoch+1, index, predicted_summary))

print('model trained')
model.save('data/models/glove_new_model_05.h5')
print('model saved')

with open('data/models/glove_new_model_tokenizer_05.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('data/models/glove_new_model_wordindex_05.pkl', 'wb') as f:
    pickle.dump(word_index, f)
