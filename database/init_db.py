import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "students.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            parental_level_of_education TEXT,
            lunch TEXT,
            test_preparation_course TEXT,
            math_score INTEGER,
            reading_score INTEGER,
            writing_score INTEGER,
            dropout_prediction TEXT
        );
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database created successfully!")
