"""
Name: Timothy James Duffy, Kevin Falconett
File: summarizer.py
Class: CSc 483; Spring 2023
Project: TextSummarizer

Generates a summary using the model in the config.
"""

import sys
import pickle
import numpy as np
from database import *
import tensorflow as tf
from preprocessor import Preprocessor
from keras.utils import pad_sequences
from config import MAX_ARTICLE_LENGTH, MAX_SUMMARY_LENGTH, DATABASE_NAME, MODEL_PATH, TOKENIZER_PATH


def generate_summary(article):
    """Generates a summary from a plain text article string."""
    # Preprocess the article.
    preprocessor = Preprocessor(DATABASE_NAME)
    article = preprocessor.process_doc_text(article)

    # Filter tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load the saved model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the tokenizer and max sequence lengths from disk
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    # Tokenize the article and pad it to max length.
    article_tokens = tokenizer.texts_to_sequences([article])
    article_input = pad_sequences(article_tokens, maxlen=MAX_ARTICLE_LENGTH)

    # Generate the summary, convert encoding to int indexes, and then to English text.
    predicted_summary = model.predict([article_input, np.zeros((1, MAX_SUMMARY_LENGTH))], verbose=0)[0]
    predicted_summary = np.argmax(predicted_summary, axis=1)
    predicted_summary = ' '.join(tokenizer.index_word[i] for i in predicted_summary if i > 0)

    # Strip out special tokens and whitespace.
    predicted_summary = postprocess_summary(predicted_summary)

    return predicted_summary


def generate_summary_by_id(doc_id):
    """Generates a summary for a document in the database given the id."""
    database = Database(DATABASE_NAME)
    article = database.get_doc_by_id(doc_id)[0]
    summary = generate_summary(article)

    return summary


def postprocess_summary(summary):
    # Strip out special tokens and whitespace.
    summary = summary.replace('<start>', '')
    summary = summary.replace('<pad>', '')
    summary = summary.replace('<OOV>', '')
    summary = summary.replace('<end>', '').strip()

    return summary


def main():
    # Accepts 1 argument: the path to the file. Run from terminal using 'python summarizer.py example_file_path.txt'
    article_path = sys.argv[0]

    # Read the file contents.
    with open(article_path, 'r') as file:
        article_text = file.read()

    # Generate and print the summary.
    summary = generate_summary(article_text)
    print('Generated summary:')
    print(summary)


if __name__ == '__main__':
    main()
