"""
Name: Timothy James Duffy, Kevin Falconett
File: preprocessor.py
Class: CSc 483; Spring 2023
Project: TextSummarizer

The preprocessor class provides methods to preprocess data and add an entire dataset to a database.
"""

import re
from database import *


class Preprocessor:
    def __init__(self, db_name):
        self.db_name = db_name

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
        """Processes the dataset and adds it to the database."""
        database = Database(self.db_name)

        # Iterate through each file in the given path.
        for i, filename in enumerate(os.listdir(path)):
            # Open the file and split it into text, summary.
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                doc = f.read()
            doc_split = doc.split('@highlight', 1)

            # Store the filename as doc_id (strip extension).
            doc_id = os.path.splitext(filename)[0]

            # Skip if the article and/or summary is missing.
            if len(doc_split) < 2:
                print("Skipping Document #{}. Missing article/summary in raw data. ID: {}".format(i, doc_id))
                continue

            # Process the raw document text and summary.
            doc_text = self.process_doc_text(doc_split[0])
            doc_summary = self.process_doc_summary(doc_split[1])

            # Skip any articles that are missing text or a summary after processing.
            if not doc_text or not doc_summary:
                print("Skipping Document #{}. Missing article/summary after processing. ID: {}".format(i, doc_id))
                continue

            database.add_doc_to_db(doc_id, doc_text, doc_summary)
            print("Database added Document #{}".format(i))
