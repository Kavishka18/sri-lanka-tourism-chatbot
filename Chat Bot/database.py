import sqlite3
import re

def clean_text(text):
    """Clean input text: lowercase, remove punctuation."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuation
    return text

class ChatBotDB:
    def __init__(self, db_name="chatbot.db"):
        try:
            self.connection = sqlite3.connect(db_name, check_same_thread=False)
            self.cursor = self.connection.cursor()
            self.create_tables()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize database: {e}")

    def create_tables(self):
        """Creates tables for storing chatbot responses and retraining counter."""
        try:
            # Responses table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE,
                    answer TEXT
                )
            """)
            # Counter table for tracking new responses
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrain_counter (
                    id INTEGER PRIMARY KEY,
                    new_responses INTEGER DEFAULT 0
                )
            """)
            # Initialize counter if not exists
            self.cursor.execute("INSERT OR IGNORE INTO retrain_counter (id, new_responses) VALUES (1, 0)")
            self.connection.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Failed to create tables: {e}")

    def insert_response(self, question, answer):
        """Insert or update a response in the database and increment counter."""
        question = clean_text(question)
        answer = answer.strip() if answer else ""
        
        # Validate inputs
        if not question or len(question) > 500:
            raise ValueError("Question must be non-empty and less than 500 characters")
        if not answer or len(answer) > 1000:
            raise ValueError("Answer must be non-empty and less than 1000 characters")
        
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO responses (question, answer) VALUES (?, ?)
            """, (question, answer))
            # Increment new_responses counter
            self.cursor.execute("UPDATE retrain_counter SET new_responses = new_responses + 1 WHERE id = 1")
            self.connection.commit()
        except sqlite3.IntegrityError as e:
            raise RuntimeError(f"Database integrity error: {e}")
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Database operation error: {e}")

    def get_response(self, question):
        """Retrieve a response from the database."""
        question = clean_text(question)
        if not question:
            return None
        try:
            self.cursor.execute("SELECT answer FROM responses WHERE question = ?", (question,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Database query error: {e}")

    def get_all_questions(self):
        """Returns all learned questions."""
        try:
            self.cursor.execute("SELECT question FROM responses")
            return [row[0] for row in self.cursor.fetchall()]
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Database query error: {e}")

    def get_all_responses(self):
        """Returns all question-answer pairs for retraining."""
        try:
            self.cursor.execute("SELECT question, answer FROM responses")
            return [(row[0], row[1]) for row in self.cursor.fetchall()]
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Database query error: {e}")

    def get_retrain_counter(self):
        """Returns the number of new responses since last retraining."""
        try:
            self.cursor.execute("SELECT new_responses FROM retrain_counter WHERE id = 1")
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Database query error: {e}")

    def reset_retrain_counter(self):
        """Resets the new responses counter after retraining."""
        try:
            self.cursor.execute("UPDATE retrain_counter SET new_responses = 0 WHERE id = 1")
            self.connection.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Database query error: {e}")

    def close_connection(self):
        """Close the database connection."""
        try:
            if self.connection:
                self.connection.close()
        except sqlite3.Error as e:
            print(f"Error closing database connection: {e}")