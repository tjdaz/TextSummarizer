import sqlite3


class Database:

    def __init__(self, name):
        self.path = 'data/database/' + name + '.db'
        self._create_database()

    def _create_database(self):
        """Creates an sqlite database for documents."""
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()

        # Create a table in the database for the document data if one does not exist.
        cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                          (id INTEGER PRIMARY KEY,
                           doc_id TEXT,
                           doc_text TEXT,
                           doc_summary TEXT,
                           doc_text_vector BLOB,
                           doc_summary_vector BLOB)''')

        print('Database successfully created at: ' + self.path)

        # Save the table and close the connection.
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
