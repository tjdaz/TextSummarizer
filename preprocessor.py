import os
import re
import sqlite3
import torch
from transformers import BertModel, BertTokenizer


class Preprocessor:

    # Constants for the number of tokens.
    MAX_TEXT_TOKENS = 512
    MAX_SUMMARY_TOKENS = 512

    def __init__(self, db_path):
        self.db_path = db_path
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def process_doc_text(self, text):
        """Processes and returns a dataset document's text."""
        # Remove the (CNN) or other news source tagline from the first line.
        text = re.sub(r'(?i)^\s*.*?\(.*?\)\s*-*\s*', '', text)
        text = re.sub(r'(?i).*contributed to this report.*(\r\n|\r|\n)?', '', text)
        text = re.sub(r'(?i)^\\s*READ MORE:.*(\r\n|\r|\n)?', '', text)
        text = text.strip()

        return text

    def process_doc_summary(self, summary):
        """Processes and returns a dataset document's summary."""
        summary = re.sub(r'^\s+', '', summary, count=1)
        summary = re.sub(r'@highlight', '', summary)
        summary = re.sub(r'\n+', '. ', summary)
        summary = re.sub(r'NEW: ', '', summary)
        summary = summary + ". "
        summary = summary.strip()
        return summary

    def process_dataset(self, path):
        """
        Processes each article in a directory and adds it to the database.
        The function is kind of long and ugly, but it is optimized to process large datasets "quickly".
        """
        if self.use_cuda:
            # Use GPU if CUDA is available.
            self.model = self.model.cuda()

        # Establish a connection to the database and create a cursor.
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # Prepared SQL insert statement.
        sql_insert = "INSERT INTO documents (doc_id, doc_text, doc_summary, doc_text_vector, doc_summary_vector) " \
                     "VALUES (?, ?, ?, ?, ?)"

        index = 0

        # Iterate through each file in the given path.
        for filename in os.listdir(path):
            # Open the file and split it into text, summary.
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                doc = f.read()
            doc_split = doc.split('@highlight', 1)

            # Store the filename as doc_id (strip extension).
            doc_id = os.path.splitext(filename)[0]

            # Skip if the article and/or summary is missing.
            if len(doc_split) < 2:
                print("Skipping Document #" + str(index) + ". Missing article/summary in raw data. ID: " + doc_id)
                continue

            # Process the raw document text and summary.
            doc_text = self.process_doc_text(doc_split[0])
            doc_summary = self.process_doc_summary(doc_split[1])

            # Skip any articles that are missing text or a summary after processing.
            if not doc_text or not doc_summary:
                print("Skipping Document #" + str(index) + ". Missing article/summary after processing. ID: " + doc_id)
                continue

            # Get tokens for the doc text and summary, adds CLS, SEP tags.
            doc_text_tokens = self.tokenizer.encode(doc_text, add_special_tokens=True, truncation=True,
                                                    max_length=Preprocessor.MAX_TEXT_TOKENS)
            doc_summary_tokens = self.tokenizer.encode(doc_summary, add_special_tokens=True, truncation=True,
                                                       max_length=Preprocessor.MAX_SUMMARY_TOKENS)

            # Create and store vectors for the doc text and summary.
            with torch.no_grad():
                doc_text_tensor = torch.tensor(doc_text_tokens).unsqueeze(0)
                doc_summary_tensor = torch.tensor(doc_summary_tokens).unsqueeze(0)

                if self.use_cuda:
                    # Move tensor to the GPU if CUDA can be used.
                    doc_text_tensor = doc_text_tensor.cuda()
                    doc_summary_tensor = doc_summary_tensor.cuda()

                # Get vector using the model.
                doc_text_vector = self.model(doc_text_tensor)[1]
                doc_summary_vector = self.model(doc_summary_tensor)[1]

                if self.use_cuda:
                    # Move the vector back to the CPU if CUDA was used.
                    doc_text_vector = doc_text_vector.cpu().numpy()
                    doc_summary_vector = doc_summary_vector.cpu().numpy()
                else:
                    doc_text_vector = doc_text_vector.cpu().numpy()
                    doc_summary_vector = doc_summary_vector.cpu().numpy()
            # Add the document to the database.
            sql_data = (doc_id, doc_text, doc_summary, doc_text_vector.tobytes(), doc_summary_vector.tobytes())
            cursor.execute(sql_insert, sql_data)
            connection.commit()
            print("Database added Document #" + str(index))
            index += 1

        # Close the connection to the database.
        connection.close()

    def add_doc_to_db(self, doc_id, doc_text, doc_summary, doc_text_vector, doc_summary_vector):
        """Adds a document into the database."""
        # Establish a connection to the database and create a cursor.
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        # Prepared SQL insert statement.
        sql_insert = "INSERT INTO documents (doc_id, doc_text, doc_summary, doc_text_vector, doc_summary_vector) " \
                     "VALUES (?, ?, ?, ?, ?)"

        # Add the document to the database.
        sql_data = (doc_id, doc_text, doc_summary, doc_text_vector.tobytes(), doc_summary_vector.tobytes())
        cursor.execute(sql_insert, sql_data)

        # Save the article to the db and close the connection.
        connection.commit()
        connection.close()

    def get_tokens(self, text, max_tokens):
        """Tokenizes and return the text using the BertTokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens)

    def get_vector(self, text):
        """Creates and returns a vector for the given tokens."""
        # Get tokens for the doc text and summary, adds CLS, SEP tags.
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True,
                                                max_length=Preprocessor.MAX_TEXT_TOKENS)

        # Create and store vectors for the doc text and summary.
        with torch.no_grad():
            text_tensor = torch.tensor(tokens).unsqueeze(0)

            # Get vector using the model.
            text_vector = self.model(doc_text_tensor)[1]

            text_vector = doc_text_vector.cpu().numpy()
            return text_vector
