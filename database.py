"""
Name: Timothy James Duffy, Kevin Falconett
File: database.py
Class: CSc 483; Spring 2023
Project: TextSummarizer

The database class provides methods to create a database, as well as insert and retrieve data.
"""

import os
import random
import sqlite3


class Database:
    def __init__(self, name):
        """Creates a Database object and a sqlite3 database with the given name."""
        self.path = 'data/database/' + name + '.db'
        self._create_database()

    def _create_database(self):
        """Creates an sqlite3 database for documents."""
        # Exit if the database already exists.
        if os.path.exists(self.path):
            return

        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Create a table in the database for the document data if one does not exist.
        cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                          (id INTEGER PRIMARY KEY,
                           doc_id TEXT,
                           doc_text TEXT,
                           doc_summary TEXT)''')

        # Save the table and close the connection.
        print('Database successfully created at: ' + self.path)
        connection.commit()
        connection.close()

    def add_doc_to_db(self, doc_id, doc_text, doc_summary):
        """Adds a document into the database."""
        # Establish a connection to the database and create a cursor.
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Insert the document with doc_id, doc_text, doc_summary.
        sql_insert = "INSERT INTO documents (doc_id, doc_text, doc_summary) VALUES (?, ?, ?)"
        sql_data = (doc_id, doc_text, doc_summary)
        cursor.execute(sql_insert, sql_data)

        # Save changes and close the connection.
        connection.commit()
        connection.close()

    def get_size(self):
        """Gets the number of documents stored in the database."""
        # Establish a connection to the database and create a cursor.
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Get number of documents (rows) in the database.
        cursor.execute("SELECT COUNT(*) FROM documents")
        size = cursor.fetchone()[0]

        # Close the connection.
        connection.close()

        return size

    def get_data(self, n, offset=0):
        """Returns a list of n documents as (article, summary)."""
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Get number of documents (rows) in the database.
        cursor.execute("SELECT doc_text, doc_summary FROM documents LIMIT ? OFFSET ?", (n, offset))
        docs = cursor.fetchall()

        # Save the table and close the connection.
        connection.commit()
        connection.close()

        return docs

    def get_random_doc(self):
        """Gets a random document from the database."""
        # Get a random id.
        rand_id = random.randrange(0, self.get_size())

        # Establish a connection to the database and create a cursor.
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Get number of documents (rows) in the database.
        cursor.execute("SELECT * FROM documents WHERE id=? LIMIT 1", (rand_id,))
        doc = cursor.fetchone()

        # Close the connection.
        connection.close()
        return doc

    def get_doc_by_id(self, doc_id):
        """Gets a document from the database by id."""
        # Establish a connection to the database and create a cursor.
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Get number of documents (rows) in the database.
        cursor.execute("SELECT * FROM documents WHERE id=? LIMIT 1", (doc_id,))
        doc = cursor.fetchone()

        # Close the connection.
        connection.close()
        return doc
