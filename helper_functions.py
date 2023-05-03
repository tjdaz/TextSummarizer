import sqlite3
import numpy as np
from torchtext.vocab import GloVe
import random
import torch

# Set the path to the database.
db_path = 'data/database/documents.db'

# Load the glove.6B.100d model.
glove_model = GloVe(name='6B', dim=300)

# Define device for tensor processing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_random_doc(path='data/database/documents.db'):
    """Gets the number of documents stored in the database."""
    # Get a random id.
    rand_id = random.randrange(0, 92400)

    # Establish a connection to the database and create a cursor.
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    # Get number of documents (rows) in the database.
    cursor.execute("SELECT * FROM documents WHERE id=? LIMIT 1", (rand_id,))
    doc = cursor.fetchone()

    # Close the connection.
    connection.close()
    return doc

def get_doc_by_id(doc_id, path='data/database/documents.db'):
    """Gets the number of documents stored in the database."""
    # Establish a connection to the database and create a cursor.
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    # Get number of documents (rows) in the database.
    cursor.execute("SELECT * FROM documents WHERE id=? LIMIT 1", (doc_id,))
    doc = cursor.fetchone()

    # Close the connection.
    connection.close()
    return doc

def print_tensor_shapes(path='data/database/documents.db'):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    cursor.execute("SELECT input_tensor, label_tensor FROM documents LIMIT 500")
    docs = cursor.fetchall()
    for doc in docs:
        print(np.frombuffer(doc[0], dtype=np.float32).shape)  # input_tensor
        print(np.frombuffer(doc[1], dtype=np.float32).shape)  # label_tensor
    connection.close()


def bytes_to_tensor(byte_tensor):
    return torch.from_numpy(np.frombuffer(byte_tensor, dtype=np.float32)).reshape(600, 300).cuda()

def bytes_to_numpy(byte_tensor):
    return np.array(np.frombuffer(byte_tensor, dtype=np.float32).reshape(600, 300))


def decode_numpy(tensor):
    # Load the glove.6B.100d model.
    glove_model = GloVe(name='6B', dim=300)

    # Move glove_model.vectors to GPU.
    glove_model.vectors = glove_model.vectors.cuda()

    # Convert numpy array to PyTorch tensor and move to GPU.
    tensor = torch.from_numpy(tensor).cuda()

    words_back = []
    for embedding in tensor:
        # Find the word with the closest embedding.
        distances = ((glove_model.vectors - embedding) ** 2).sum(dim=1)
        word_idx = distances.argmin().item()
        word = glove_model.itos[word_idx]
        words_back.append(word)
    # Join the words to form the sentence.
    sentence_back = " ".join(words_back)
    #print(sentence_back)
    return sentence_back

def decode_tensor(tensor):
    # Load the glove.6B.100d model.
    glove_model = GloVe(name='6B', dim=300)

    # Move glove_model.vectors to GPU.
    glove_model.vectors = glove_model.vectors.cuda()

    words_back = []
    for x in tensor:
        for embedding in x:
            # Find the word with the closest embedding.
            distances = ((glove_model.vectors - embedding) ** 2).sum(dim=1)
            word_idx = distances.argmin().item()
            word = glove_model.itos[word_idx]
            words_back.append(word)
    # Join the words to form the sentence.
    sentence_back = " ".join(words_back)
    #print(sentence_back)
    return sentence_back
