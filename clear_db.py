import sqlite3
import os

DB_FILENAME = "processed_ids.sqlite"
TABLE_NAME = "processed_assets"

def clear_processed_ids():
    if not os.path.exists(DB_FILENAME):
        print(f"Database file '{DB_FILENAME}' not found. Nothing to clear.")
        return

    conn = None
    try:
        print(f"Connecting to '{DB_FILENAME}'...")
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        
        # Drop the existing table first to ensure schema update
        print(f"Dropping table '{TABLE_NAME}' if it exists...")
        cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        
        # Recreate the table with the correct schema
        print(f"Recreating table '{TABLE_NAME}' with current schema...")
        cur.execute("""
            CREATE TABLE processed_assets (
                asset_id TEXT,
                matched_keyword TEXT,
                similarity_score REAL,
                target_album_id TEXT,
                target_album_name TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Delete statement is no longer needed as DROP/CREATE clears it
        # print(f"Deleting all entries from table '{TABLE_NAME}'...")
        # cur.execute(f"DELETE FROM {TABLE_NAME}")
        # conn.commit()
        
        # Optional: Vacuum to reclaim disk space (might not be needed for just deletes)
        # print("Vacuuming database...")
        # conn.execute("VACUUM")
        
        print("Successfully cleared processed IDs table.")
        
    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    print("--- Clearing Processed IDs Database ---")
    clear_processed_ids()
    print("-------------------------------------") 