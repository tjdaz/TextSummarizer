import sys
import pickle
import numpy as np
from database import *
import tensorflow as tf
from keras.utils import pad_sequences
from build import MAX_ARTICLE_LENGTH, MAX_SUMMARY_LENGTH, DATABASE_NAME


def generate_summary(article):
    # Filter tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load the saved model
    model = tf.keras.models.load_model('data/models/model.h5')

    # Load the tokenizer and max sequence lengths from disk
    with open('data/models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Tokenize the article and pad it to max length.
    article_tokens = tokenizer.texts_to_sequences([article])
    article_input = pad_sequences(article_tokens, maxlen=MAX_ARTICLE_LENGTH)

    # Generate the summary, convert encoding to int indexes, and then to English text.
    predicted_summary = model.predict([article_input, np.zeros((1, MAX_SUMMARY_LENGTH))], verbose=0)[0]
    predicted_summary = np.argmax(predicted_summary, axis=1)
    predicted_summary = ' '.join(tokenizer.index_word[i] for i in predicted_summary if i > 0)

    return predicted_summary


def generate_summary_by_id(doc_id):
    database = Database(DATABASE_NAME)
    article = database.get_doc_by_id(doc_id)[0]
    summary = generate_summary(article)
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
