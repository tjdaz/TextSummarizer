import metrics
import neural_network
from preprocessor import *


# Constants for the raw data paths.
TOKENIZER_PATH = 'data/models/tokenizer.pkl'
MODEL_PATH = 'data/models/model.h5'
DATABASE_NAME = 'documents'
RAW_DATA_PATH = 'data/raw/'
MAX_ARTICLE_LENGTH = 1621
MAX_SUMMARY_LENGTH = 70
MIN_DOCUMENTS = 1500


def main():
    # Do not run this if you have a model already created
    database = Database('documents_cnn_2k')
    preprocessor = Preprocessor(database.path)

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
