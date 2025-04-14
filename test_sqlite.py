import sqlite3
import sys

db_file = "processed_ids.sqlite"
conn = None
cur = None

print(f"Attempting to connect to: {db_file}")
try:
    # Try connecting with a timeout and exclusive lock attempt (might reveal locking issues)
    conn = sqlite3.connect(db_file, timeout=10) # 10 second timeout
    conn.isolation_level = 'EXCLUSIVE' # Try to get an exclusive lock immediately
    conn.execute('BEGIN EXCLUSIVE') # Explicitly start exclusive transaction

    cur = conn.cursor()
    print("Connection successful, attempting to write...")

    # Try creating table just in case (should succeed harmlessly if exists)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_assets (
            asset_id TEXT,
            matched_keyword TEXT,
            similarity_score REAL,
            target_album_id TEXT,
            target_album_name TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Attempt to insert a dummy row
    cur.execute("INSERT INTO processed_assets (asset_id, matched_keyword, similarity_score, target_album_id, target_album_name) VALUES (?, ?, ?, ?, ?)",
                ('test_id', 'test_keyword', 0.99, 'test_album_id', 'test_album_name'))

    print("Write successful, attempting to commit...")
    conn.commit()
    print("Commit successful!")

    # Clean up dummy row
    print("Attempting to delete test row...")
    cur.execute("DELETE FROM processed_assets WHERE asset_id = ?", ('test_id',))
    conn.commit()
    print("Test row deleted successfully.")

except sqlite3.OperationalError as e:
    print(f"SQLite OperationalError: {e}", file=sys.stderr)
    if "readonly database" in str(e):
            print(">>> Encountered the 'readonly database' error.", file=sys.stderr)
    elif "locked" in str(e):
            print(">>> Encountered a 'database is locked' error.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    if conn:
            conn.rollback() # Rollback on unexpected error
    sys.exit(1)
finally:
    print("Closing connection...")
    if cur:
        cur.close()
    if conn:
        conn.close()
    print("Connection closed.")
