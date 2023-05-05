"""
Name: Timothy James Duffy, Kevin Falconett
File: build.py
Class: CSc 483; Spring 2023
Project: TextSummarizer

Builds the database from raw files, builds and trains the neural network, and runs metrics on the model.
"""

import metrics
import neural_network
from preprocessor import *
from config import MIN_DOCUMENTS, RAW_DATA_PATH, DATABASE_NAME


def main():
    # Filter tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Do not run this if you have a model already created
    database = Database(DATABASE_NAME)
    preprocessor = Preprocessor(DATABASE_NAME)

    # Process data and add it to the database if database does not have documents.
    if database.get_size() < MIN_DOCUMENTS:
        preprocessor.process_dataset(RAW_DATA_PATH)

    # Files have been added to the database.
    print(str(database.get_size()) + ' documents have been added to the database.')
    print('Each document entry includes an id, doc_id, doc_text, and doc_summary.')

    # Build and train the neural network.
    neural_network.main()

    # Run metrics on the trained neural network.
    metrics.main()


if __name__ == '__main__':
    main()
