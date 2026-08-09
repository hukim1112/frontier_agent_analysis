import sqlite3
import json
import time

def populate():
    conn = sqlite3.connect('app/database/episodic.db')
    cursor = conn.cursor()

    # Ensure tables exist
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        summary TEXT,
        keywords TEXT,
        message_count INTEGER DEFAULT 0,
        created_at REAL,
        updated_at REAL
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT,
        tool_name TEXT,
        created_at REAL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
        summary, keywords
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content
    );
    """)

    # Clear existing to avoid duplicates
    cursor.execute("DELETE FROM sessions WHERE session_id = 'session_001_initial'")
    cursor.execute("DELETE FROM messages WHERE session_id = 'session_001_initial'")

    # Load session_001_initial.jsonl line by line
    dialogue_history = []
    with open('artifacts/logs/session_001_initial.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                dialogue_history.extend(data.get('dialogue_history', []))
    
    if not dialogue_history:
        print("No dialogue history found!")
        return

    # Predefined summary and keywords
    summary = "The user introduced themselves as Park Hyun-woo (and later Jade), a Python developer. They discussed their project's technology stack, which includes FastAPI and PostgreSQL 15, and their preference for pytest as the unit testing framework."
    keywords = ["FastAPI", "PostgreSQL", "pytest", "Python", "Jade", "Park Hyun-woo", "session_001"]

    now = time.time()
    try:
        cursor.execute("""
            INSERT INTO sessions (session_id, summary, keywords, message_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ('session_001_initial', summary, json.dumps(keywords), len(dialogue_history), now, now))
        rowid = cursor.lastrowid
        print("Inserted session, rowid:", rowid)
    except Exception as e:
        print("Error inserting session:", e)
        raise e

    # Delete existing FTS entry if any
    try:
        cursor.execute("DELETE FROM sessions_fts WHERE rowid = ?", (rowid,))
    except Exception as e:
        print("Error deleting from sessions_fts:", e)

    try:
        # Insert into sessions_fts
        cursor.execute("""
            INSERT INTO sessions_fts (rowid, summary, keywords)
            VALUES (?, ?, ?)
        """, (rowid, summary, json.dumps(keywords)))
        print("Inserted sessions_fts")
    except Exception as e:
        print("Error inserting sessions_fts:", e)
        raise e

    # Insert messages
    for msg in dialogue_history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        try:
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, ('session_001_initial', role, content, now))
            msg_id = cursor.lastrowid
        except Exception as e:
            print("Error inserting message:", e)
            raise e

        # Delete existing messages_fts entry if any
        try:
            cursor.execute("DELETE FROM messages_fts WHERE rowid = ?", (msg_id,))
        except Exception as e:
            pass

        # Insert into messages_fts
        if content:
            try:
                cursor.execute("""
                    INSERT INTO messages_fts (rowid, content)
                    VALUES (?, ?)
                """, (msg_id, content))
            except Exception as e:
                print("Error inserting messages_fts:", e)
                pass

    conn.commit()
    conn.close()
    print("Successfully populated session_001_initial into episodic.db!")

if __name__ == '__main__':
    populate()
