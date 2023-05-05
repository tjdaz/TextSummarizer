"""
Name: Timothy James Duffy, Kevin Falconett
File: neural_network.py
Class: CSc 483; Spring 2023
Project: TextSummarizer

Builds and trains a seq2seq neural network model for article summarization.
"""

# Filter tensorflow warnings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
from database import *
import tensorflow as tf
from keras.utils import pad_sequences
from summarizer import postprocess_summary
from keras.preprocessing.text import Tokenizer
from config import MAX_ARTICLE_LENGTH, MAX_SUMMARY_LENGTH, DATABASE_NAME, MODEL_PATH, TOKENIZER_PATH

# Constants for the neural network model.
EMBEDDING_DIM = 300
BATCH_SIZE = 16
TEST_INDEX = 0
NUM_DOCS = 64
EPOCHS = 10

# Instantiate the tokenizer.
tokenizer = Tokenizer(filters='', oov_token='<OOV>', lower=True)


def load_data():
    """Loads articles and summaries from the database."""
    # Load CNN articles from database.
    print('Loading training data...')
    database = Database(DATABASE_NAME)
    articles, summaries = zip(*database.get_data(NUM_DOCS))

    return articles, summaries


def tokenize(articles, summaries):
    """Tokenizes ands pads articles and summaries."""
    # Tokenize the training articles and summaries.
    print('Tokenizing training data...')
    tokenizer.fit_on_texts([article for article in articles] + [summary for summary in summaries])
    # Convert the token sequences to their integer representation.
    tokenizer.fit_on_texts(['<start> ' + article[:MAX_ARTICLE_LENGTH] + ' <end>' for article in articles] +
                           ['<start> ' + summary + ' <end>' for summary in summaries])
    article_sequences = tokenizer.texts_to_sequences(['<start> ' + article + ' <end>' for article in articles])
    summary_sequences = tokenizer.texts_to_sequences(['<start> ' + summary + ' <end>' for summary in summaries])

    # Padding the articles and summaries to max length.
    print('Padding/truncating training data...')

    article_input = pad_sequences(article_sequences, maxlen=MAX_ARTICLE_LENGTH)
    summary_input = pad_sequences(summary_sequences, maxlen=MAX_SUMMARY_LENGTH)

    return article_input, summary_input


def build_embedding_index():
    """Builds an embeddings index from the GloVe 300dim dataset."""
    # Load the pre-trained GloVe embeddings from file.
    print('Loading GloVe embeddings...')
    embedding_file = 'data/glove/glove.6B.300d.txt'
    embedding_index_file = 'data/glove/glove.6B.300d.txt.pkl'

    # Open the saved embedding index of all GloVe embeddings if it exists.
    if os.path.isfile(embedding_index_file):
        print('Loading the embedding index from file...')
        with open(embedding_index_file, 'rb') as f:
            embedding_index = pickle.load(f)

    # Otherwise, create the embedding index.
    else:
        print('Creating embedding index...')
        embedding_index = {}
        with open(embedding_file, encoding='utf8') as f:
            for line in f:
                vals = line.split()  # Get each word embedding.
                word = vals[0]       # Get each word.
                embedding = np.asarray(vals[1:], dtype='float32')
                embedding_index[word] = embedding  # Create mapping.
        # Write to file.
        with open(embedding_index_file, 'wb') as f:
            pickle.dump(embedding_index, f)
    return embedding_index


def get_num_words(embedding_index_len):
    """Returns the number of words in the word/embedding index (whichever is smaller)."""
    word_index = tokenizer.word_index
    num_words = min(len(word_index) + 1, embedding_index_len)
    return num_words


def build_embedding_matrix(embedding_index):
    """Builds a matrix of embeddings from the words in the embedding index."""
    print('Creating embedding matrix...')
    word_index = tokenizer.word_index
    num_words = min(len(word_index) + 1, len(embedding_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def build_neural_network(embedding_matrix, num_words):
    """Builds the seq2seq neural network."""
    # Build the seq2seq neural network model.
    print('Building and compiling neural network model...')

    # Create neural network input layers.
    article_input_layer = tf.keras.layers.Input(shape=(MAX_ARTICLE_LENGTH,))
    summary_input_layer = tf.keras.layers.Input(shape=(MAX_SUMMARY_LENGTH,))

    # Create neural network embedding layers.
    embedding_layer = tf.keras.layers.Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
    article_embedding = embedding_layer(article_input_layer)
    summary_embedding = embedding_layer(summary_input_layer)

    # Create neural network LSTM layers.
    article_lstm_layer = tf.keras.layers.LSTM(100)(article_embedding)
    summary_lstm_layer = tf.keras.layers.LSTM(100)(summary_embedding)

    # Create neural network merged and output layers.
    merged_layer = tf.keras.layers.concatenate([article_lstm_layer, summary_lstm_layer])
    output_layer = tf.keras.layers.Dense(MAX_SUMMARY_LENGTH * num_words, activation='tanh')(merged_layer)
    output_layer = tf.keras.layers.Reshape((MAX_SUMMARY_LENGTH, num_words))(output_layer)

    # Create and compile neural network model.
    model = tf.keras.models.Model(inputs=[article_input_layer, summary_input_layer], outputs=output_layer)
    model.compile(loss='cosine_similarity', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, article_input, summary_input, num_words):
    """Trains the model on the training data."""
    print('Training model...')
    for epoch in range(EPOCHS):
        for i in range(0, len(article_input), BATCH_SIZE):
            # Convert the summary input into tf.one_hot binary vectors representing the target output.
            bin_vector = tf.one_hot(summary_input[i:i + BATCH_SIZE], num_words)

            # Train the model on batches of article and summary inputs, and binary vectors.
            loss, acc = model.train_on_batch([article_input[i:i+BATCH_SIZE], summary_input[i:i+BATCH_SIZE]], bin_vector)

        # Predict the summary for a single article (article at TEST_INDEX).
        article = article_input[TEST_INDEX]
        article = np.expand_dims(article, axis=0)  # Adds batch dimension to the article array.
        predicted_summary = model.predict([article, summary_input[TEST_INDEX:TEST_INDEX+1]])[0]  # Generate the summary.
        predicted_summary = np.argmax(predicted_summary, axis=1)  # Convert binary vector to integer indexes.
        predicted_summary = ' '.join(tokenizer.index_word[i] for i in predicted_summary if i > 0)  # int index => word

        # Strip out special tokens and whitespace.
        predicted_summary = postprocess_summary(predicted_summary)

        # Print epoch loss and accuracy metrics, as well as the progress of one of the summaries.
        print('Epoch {}: loss = {}, accuracy = {}'.format(epoch+1, loss, acc))
        print('Predicted Summary (Article {}):\n{}\n'.format(TEST_INDEX, predicted_summary))

    # Save the model to disk.
    print('Saving model...')
    model.save(MODEL_PATH)

    # Save the tokenizer to disk.
    print('Saving tokenizer...')
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Training is complete.
    print('Training complete.')


def main():
    # Load training articles and summaries from database.
    articles, summaries = load_data()

    # Tokenize and pad the articles and summaries.
    article_input, summary_input = tokenize(articles, summaries)

    # Build embedding index, matrix and get the number of words in training data.
    embedding_index = build_embedding_index()
    embedding_index_len = len(embedding_index)
    num_words = get_num_words(embedding_index_len)
    embedding_matrix = build_embedding_matrix(embedding_index)

    # Build and train the neural network model on the training data.
    model = build_neural_network(embedding_matrix, num_words)
    train_model(model, article_input, summary_input, num_words)


if __name__ == '__main__':
    main()
