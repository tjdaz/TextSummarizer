from database import *
from preprocessor import *

MIN_DOCUMENTS = 90000

# Constants for the raw data paths.
TRAIN_DATA_PATH = 'data/raw/train'
TEST_DATA_PATH = 'data/raw/test'
VAL_DATA_PATH = 'data/raw/val'


def main():
    database = Database('documents')
    preprocessor = Preprocessor(database.path)
    if database.get_size() < MIN_DOCUMENTS:
        preprocessor.process_dataset(TRAIN_DATA_PATH)
        preprocessor.process_dataset(TEST_DATA_PATH)
        preprocessor.process_dataset(VAL_DATA_PATH)

    print(str(database.get_size()) + ' documents have been added to the database.')
    print('Each document includes an id, text, summary, text vector, and summary vector.')


if __name__ == '__main__':
    main()
